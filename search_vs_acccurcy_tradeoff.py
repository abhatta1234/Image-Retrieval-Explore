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


class FeatureIndexer:
    """
    Extracts features for images, builds FAISS indices (Flat and IVF) for both inner-product and L2 metrics,
    and performs searches to compare timings against brute-force cosine similarity.
    """
    def __init__(self,
                 intermediate_dir: Path,
                 model_name: str = 'resnet18',
                 batch_size: int = 32,
                 device: str = None,
                 image_list: Path = None):
        self.intermediate_dir = Path(intermediate_dir)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_list = image_list

        # Paths for intermediate files
        self.subset_list_path = self.intermediate_dir / 'subset_paths.txt'
        self.feats_path = self.intermediate_dir / 'feats.npy'

        # FAISS index paths (optional save)
        self.flat_ip_path = self.intermediate_dir / 'flat_ip.idx'
        self.ivf_ip_path = self.intermediate_dir / 'ivf_ip.idx'
        self.flat_l2_path = self.intermediate_dir / 'flat_l2.idx'
        self.ivf_l2_path = self.intermediate_dir / 'ivf_l2.idx'

        # Load model and preprocessor
        self._load_model()
        self._init_preprocessor()

    def _load_model(self):
        self.model, _ = get_feature_extractor(
            self.model_name, pretrained=True, device=self.device)
        self.model.eval()

    def _init_preprocessor(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def gather_image_paths(self, subset_percentage: float = 0.1) -> list[Path]:
        """
        Reads all image paths from a list file, samples a random subset,
        writes the subset list to disk, and returns the selected paths.
        """
        if not self.image_list:
            raise ValueError("--all_img_list must be provided to sample a subset")

        with open(self.image_list, 'r') as f:
            all_paths = [Path(line.strip()) for line in f if line.strip()]

        subset = np.random.choice(
            all_paths, max(1, int(len(all_paths) * subset_percentage)), replace=False
        ).tolist()

        with open(self.subset_list_path, 'w') as f:
            f.write("\n".join(str(p) for p in subset))

        return subset

    def extract_and_index(self, subset_percentage: float = 0.1):
        """
        Samples image paths, extracts normalized features, builds and returns
        FAISS indices and the feature matrix.
        Returns:
            subset_paths, feats, flat_ip, ivf_ip, flat_l2, ivf_l2
        """
        # Gather subset of image paths
        subset_paths = self.gather_image_paths(subset_percentage)

        # Extract features
        feats_list = []
        for i in tqdm(range(0, len(subset_paths), self.batch_size), desc="Extract features"):
            batch = subset_paths[i:i + self.batch_size]
            imgs = [self.preprocess(Image.open(p).convert('RGB')).unsqueeze(0) for p in batch]

            batch_tensor = torch.cat(imgs, dim=0).to(self.device)
            with torch.no_grad():
                raw_feats = self.model(batch_tensor)
                normed_feats = torch.nn.functional.normalize(raw_feats, p=2, dim=1).cpu().numpy()
            feats_list.append(normed_feats)

        feats = np.vstack(feats_list).astype('float32')
        np.save(self.feats_path, feats)

        N, dim = feats.shape
        nlist = int(np.sqrt(N))

        # Build IP indices
        flat_ip = faiss.IndexFlatIP(dim)
        flat_ip.add(feats)
        faiss.write_index(flat_ip, str(self.flat_ip_path))

        quant_ip = faiss.IndexFlatIP(dim)
        ivf_ip = faiss.IndexIVFFlat(quant_ip, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_ip.train(feats)
        ivf_ip.add(feats)
        faiss.write_index(ivf_ip, str(self.ivf_ip_path))

        # Build L2 indices
        flat_l2 = faiss.IndexFlatL2(dim)
        flat_l2.add(feats)
        faiss.write_index(flat_l2, str(self.flat_l2_path))

        quant_l2 = faiss.IndexFlatL2(dim)
        ivf_l2 = faiss.IndexIVFFlat(quant_l2, dim, nlist, faiss.METRIC_L2)
        ivf_l2.train(feats)
        ivf_l2.add(feats)
        faiss.write_index(ivf_l2, str(self.ivf_l2_path))

        return subset_paths, feats, flat_ip, ivf_ip, flat_l2, ivf_l2

    def extract_query_feature_single(self, path: Path) -> np.ndarray:
        """
        Extracts a normalized feature vector for a single query image.
        """
        img = Image.open(path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1).cpu().numpy()
        return feat


def search_faiss(index: faiss.Index, query_feat: np.ndarray, top_k: int):
    """
    Performs FAISS search and returns distances and indices for one query.
    """
    distances, indices = index.search(query_feat.astype('float32'), top_k)
    return distances[0], indices[0]


def brute_cosine_search(feats: np.ndarray, query_feat: np.ndarray, top_k: int):
    """
    Computes cosine similarity against a feature gallery and returns top_k results.
    """
    sims = feats.dot(query_feat.T).squeeze()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return sims[top_idx], top_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure average timing: FAISS indices vs brute-force similarity'
    )
    parser.add_argument('--intermediate_dir', required=True, type=Path,
                        help='Directory for intermediate files')
    parser.add_argument('--all_img_list', required=True, type=Path,
                        help='TXT file listing all image paths')
    parser.add_argument('--model', choices=['resnet18', 'mobilenet', 'clip'],
                        default='resnet18', help='Backbone model')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--metric', choices=['l2', 'ip'], default='ip',
                        help='Metric: l2 or inner-product')
    parser.add_argument('--k', type=int, default=5, help='Top-K for search')
    parser.add_argument('--subset_pct', type=float, default=0.1,
                        help='Subset percentage to sample')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    indexer = FeatureIndexer(
        intermediate_dir=args.intermediate_dir,
        model_name=args.model,
        batch_size=args.bs,
        device=None,
        image_list=args.all_img_list
    )

    subset, feats, flat_ip, ivf_ip, flat_l2, ivf_l2 = \
        indexer.extract_and_index(args.subset_pct)

    # Choose indices based on metric
    if args.metric == 'ip':
        flat_idx, ivf_idx = flat_ip, ivf_ip
    else:
        flat_idx, ivf_idx = flat_l2, ivf_l2

    total_times = {'extract': 0.0, 'flat': 0.0, 'ivf': 0.0, 'brute': 0.0}
    correct = 0

    for path in tqdm(subset, desc="Query images"):
        # Extract query feature
        t0 = time.perf_counter()
        q_feat = indexer.extract_query_feature_single(path)
        total_times['extract'] += time.perf_counter() - t0

        # FAISS flat search
        t0 = time.perf_counter()
        _, idx_flat = search_faiss(flat_idx, q_feat, args.k)
        total_times['flat'] += time.perf_counter() - t0

        # FAISS IVF search
        t0 = time.perf_counter()
        _, idx_ivf = search_faiss(ivf_idx, q_feat, args.k)
        total_times['ivf'] += time.perf_counter() - t0

        # Brute-force cosine
        t0 = time.perf_counter()
        _, idx_brute = brute_cosine_search(feats, q_feat, args.k)
        total_times['brute'] += time.perf_counter() - t0

        # Compute overlap between brute and IVF as correctness proxy
        correct += np.intersect1d(idx_brute, idx_ivf).size

    n = len(subset)
    logging.info("Gallery size: %d samples", n)
    logging.info("Avg extract time: %.5fs", total_times['extract'] / n)
    logging.info("Avg brute cosine time: %.5fs", total_times['brute'] / n)
    logging.info("Avg FAISS flat time: %.5fs", total_times['flat'] / n)
    logging.info("Avg FAISS IVF time: %.5fs", total_times['ivf'] / n)

    accuracy = correct / (n * args.k) * 100
    logging.info("Average overlap accuracy (IVF vs brute): %.2f%%", accuracy)


if __name__ == '__main__':
    main()
