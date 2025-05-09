'''

Experimenting with average timing for faiss index vs brute search
Given a list of images, search against both and return the average time
This can be broken down in multiple codes
But for ease of reading doing all in one

'''
import argparse
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from feature_extractor import get_feature_extractor

# Constants
IMAGE_SIZE = 256
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


'''

Given the list of image paths
Extracts features, saves to intermediate directory 
and the does the faiss indexing

'''


class FeatureIndexer:
    def __init__(self,
                 out_dir: Path,
                 model_name: str = 'resnet18',
                 batch_size: int = 32,
                 metric: str = 'ip',
                 device: str = None,
                 image_list=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.metric = metric
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_list = image_list  # list[Path] or None

        # output files
        self.feats_path = self.out_dir / 'feats.npy'
        self.intermediate_imgs_path = self.out_dir / 'img_paths.txt'
        self.flat_idx_path = self.out_dir / 'flat.idx'
        self.ivf_idx_path = self.out_dir / 'ivf.idx'

        self._load_model()
        self._init_preprocessor()

    def _load_model(self):
        self.model, _ = get_feature_extractor(
            self.model_name, pretrained=True, device=self.device
        )
        self.model.eval()

    def _init_preprocessor(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def gather_image_paths(self, subset_percentage=0.1):
        """Read image paths from file and randomly select subset_percentage of them"""
        import random
        from pathlib import Path

        if not self.image_list:
            raise ValueError("image_list path must be provided for subset indexing")

        # Read paths from the image list file
        with open(self.image_list, 'r') as f:
            paths = [Path(line.strip()) for line in f if line.strip()]

        # Select random subset
        selected = random.sample(paths, max(1, int(len(paths) * subset_percentage)))

        # Write selected paths to output file
        with open(self.imgs_path, 'w') as f:
            f.write("\n".join(str(p) for p in selected))

        return selected

    def extract_and_index(self):
        # extract
        image_paths = self._gather_image_paths()
        feats_list = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Extract features"):
            batch = image_paths[i:i+self.batch_size]
            tensors = []
            for p in batch:
                img = Image.open(p).convert('RGB')
                tensors.append(self.preprocess(img).unsqueeze(0))

            bt = torch.cat(tensors, dim=0).to(self.device)
            with torch.no_grad():
                feat = self.model(bt)
                normed = torch.nn.functional.normalize(feat, p=2, dim=1).cpu().numpy()
            feats_list.append(normed)

        feats = np.vstack(feats_list).astype('float32')
        np.save(self.feats_path, feats)

        # index
        dim = feats.shape[1]
        if self.metric == 'ip':
            flat = faiss.IndexFlatIP(dim)
            ivf_metric = faiss.METRIC_INNER_PRODUCT
        else:
            flat = faiss.IndexFlatL2(dim)
            ivf_metric = faiss.METRIC_L2

        flat.add(feats)
        faiss.write_index(flat, str(self.flat_idx_path))

        nlist = int(np.sqrt(len(feats)))
        quant = faiss.IndexFlatIP(dim) if self.metric=='ip' else faiss.IndexFlatL2(dim)
        ivf = faiss.IndexIVFFlat(quant, dim, nlist, ivf_metric)
        ivf.train(feats)
        ivf.add(feats)
        faiss.write_index(ivf, str(self.ivf_idx_path))

        return feats, flat, ivf


def extract_query_feature(path: Path, model: torch.nn.Module, preprocess: transforms.Compose,
                          device: str) -> np.ndarray:
    """Extract and L2-normalize feature vector for a single image."""
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1).cpu().numpy()
    return feat


def load_faiss_index(index_path: Path) -> faiss.Index:
    """Load FAISS index from file, raising if not found."""
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return faiss.read_index(str(index_path))


def search_faiss(index: faiss.Index, query_feat: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform FAISS search and return distances and indices."""
    distances, indices = index.search(query_feat.astype('float32'), top_k)
    return distances[0], indices[0]


def brute_cosine_search(feats: np.ndarray, query_feat: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute cosine similarity against a feature gallery and return top results."""
    sims = feats.dot(query_feat.T).squeeze()
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return sims[top_indices], top_indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Search query images against FAISS and brute-force searches'
    )
    parser.add_argument(
        '--out-dir', required=True, type=Path,
        help='Directory containing saved indexes, features, and image paths file'
    )
    parser.add_argument(
        '--model', choices=['resnet18', 'mobilenet', 'clip'],
        default='resnet18', help='Backbone model to use'
    )
    parser.add_argument(
        '--k', type=int, default=5,
        help='Number of top results to retrieve'
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for running searches and logging timings."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = get_feature_extractor(
        args.model, pretrained=True, device=device
    )
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    fo

    indexer = FeatureIndexer(
        out_dir=sub_out,
        model_name=args.model,
        batch_size=args.bs,
        metric=args.metric,
        device=device,
        image_list=subset
    )

    subset_image_paths=indexer._gather_image_paths() # using this function inside extract_index_as_well
    feats, flat_idx, ivf_idx = indexer.extract_and_index()

    total_extract = total_faiss_l2 = total_faiss_ivf = total_brute = 0.0


    for img_path in tqdm(subset_image_paths):
        start_feat = time.perf_counter()
        q_feat = extract_query_feature(img_path, model, preprocess, device)
        total_extract += time.perf_counter() - start_feat

        start_faiss_flat = time.perf_counter()
        dist_flat,indx_flat = search_faiss(flat_l2_index, q_feat, args.k)
        total_faiss_l2 += time.perf_counter() - start_faiss_flat

        start_faiss_ivf = time.perf_counter()
        dist_ivf,indx_ivf = search_faiss(ivf_index, q_feat, args.k)
        total_faiss_ivf += time.perf_counter() - start_faiss_ivf

        start_cos_brute = time.perf_counter()
        dist_brute,indx_brute = brute_cosine_search(feats, q_feat, args.k)
        total_brute += time.perf_counter() - start_cos_brute

        accuracy = np.set1diff(indx_brute,indx_ivf)/len(indx_ivf)*100

    n = len(paths)
    logging.info("Size of search gallery: %d", n)
    logging.info("Query extract: %.5fs", total_extract / n)
    logging.info("Brute cosine: %.5fs", total_brute / n)
    logging.info("Flat L2 search: %.5fs", total_faiss_l2 / n)
    logging.info("Flat IVF search: %.5fs", total_faiss_ivf / n)


if __name__ == "__main__":
    main()
