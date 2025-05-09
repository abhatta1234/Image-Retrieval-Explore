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
                 model_name: str = 'resnet18',
                 batch_size: int = 32,
                 device: str = None,
                 image_list: Path = None):

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.image_list = image_list

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
        and returns the selected paths.
        """
        if not self.image_list:
            raise ValueError("--all_img_list must be provided to sample a subset")

        with open(self.image_list, 'r') as f:
            all_paths = [Path(line.strip()) for line in f if line.strip()]

        subset = np.random.choice(
            all_paths, max(1, int(len(all_paths) * subset_percentage)), replace=False
        ).tolist()

        return subset

    def extract_and_index(self, subset_percentage: float = 0.1):
        """
        Samples image paths, extracts normalized features, and builds FAISS indices.
        Returns:
            subset_paths, feats, flat_ip, ivf_ip, flat_l2, ivf_l2
        """
        subset_paths = self.gather_image_paths(subset_percentage)
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

        N, dim = feats.shape
        nlist = int(np.sqrt(N))

        # Build IP indices
        flat_ip = faiss.IndexFlatIP(dim)
        flat_ip.add(feats)

        quant_ip = faiss.IndexFlatIP(dim)
        ivf_ip = faiss.IndexIVFFlat(quant_ip, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_ip.train(feats)
        ivf_ip.add(feats)

        # Build L2 indices
        flat_l2 = faiss.IndexFlatL2(dim)
        flat_l2.add(feats)

        quant_l2 = faiss.IndexFlatL2(dim)
        ivf_l2 = faiss.IndexIVFFlat(quant_l2, dim, nlist, faiss.METRIC_L2)
        ivf_l2.train(feats)
        ivf_l2.add(feats)

        return subset_paths, feats, flat_ip, ivf_ip, flat_l2, ivf_l2

    def extract_query_feature_single(self, path: Path) -> np.ndarray:
        """
        Extracts a normalized feature vector for a single query image.
        """
        img = Image.open(path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return torch.nn.functional.normalize(feat, p=2, dim=1).cpu().numpy()


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

    parser.add_argument('--all_img_list', required=True, type=Path,
                        help='TXT file listing all image paths')
    parser.add_argument('--model', choices=['resnet18', 'mobilenet', 'clip'],
                        default='resnet18', help='Backbone model')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--metric', choices=['l2', 'ip'], default='ip',
                        help='Metric: l2 or inner-product')
    parser.add_argument('--k', type=int, default=5, help='Top-K for search')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    indexer = FeatureIndexer(
        model_name=args.model,
        batch_size=args.bs,
        device=device,
        image_list=args.all_img_list
    )

    subset_pcts = [0.2, 0.4, 0.6, 0.8, 1.0]

    for pct in subset_pcts:
        logging.info("\n=== Experiment at subset percentage: %.1f ===", pct)
        subset, feats, flat_ip, ivf_ip, flat_l2, ivf_l2 = \
            indexer.extract_and_index(pct)

        flat_idx, ivf_idx = (flat_ip, ivf_ip) if args.metric == 'ip' else (flat_l2, ivf_l2)

        total_times = {'extract': 0.0, 'flat': 0.0, 'ivf': 0.0, 'brute': 0.0}
        correct = 0
        n = len(subset)

        for path in tqdm(subset, desc=f"Queries @ {pct:.1f}"):
            t0 = time.perf_counter()
            q_feat = indexer.extract_query_feature_single(path)
            total_times['extract'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            _, _ = search_faiss(flat_idx, q_feat, args.k)
            total_times['flat'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            _, _ = search_faiss(ivf_idx, q_feat, args.k)
            total_times['ivf'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            _, idx_brute = brute_cosine_search(feats, q_feat, args.k)
            total_times['brute'] += time.perf_counter() - t0

            # Count overlap between brute and IVF result indices
            correct += np.intersect1d(idx_brute, idx_ivf).size

        logging.info("Gallery size: %d", n)
        logging.info("Avg extract time: %.5fs", total_times['extract'] / n)
        logging.info("Avg brute time:   %.5fs", total_times['brute'] / n)
        logging.info("Avg FAISS flat:   %.5fs", total_times['flat'] / n)
        logging.info("Avg FAISS IVF:    %.5fs", total_times['ivf'] / n)
        accuracy = correct / (n * args.k) * 100
        logging.info("Overlap accuracy: %.2f%%", accuracy)

if __name__ == '__main__':
    main()
