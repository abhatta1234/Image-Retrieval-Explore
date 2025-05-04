"""Search query images against FAISS and brute-force cosine similarity."""
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


def iter_paths(paths_file: Path):
    """ Generator object to yeild next image path in case want to do this"""
    with open(paths_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Path(line)


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

    img_paths_file = args.out_dir / "img_paths.txt"
    if not img_paths_file.is_file():
        logging.error(f"Image paths file not found: {img_paths_file}")
        return

    # can lazily use generator object using iter paths functions
    # rather than loading all paths at once
    with open(img_paths_file) as f:
        paths = [line.strip() for line in f]

    flat_l2_index = load_faiss_index(args.out_dir / "flat.idx")
    ivf_index = load_faiss_index(args.out_dir / "ivf.idx")

    feats_path = args.out_dir / "feats.npy"
    if not feats_path.is_file():
        logging.error(f"Features file not found: {feats_path}")
        return
    feats = np.load(str(feats_path))

    total_extract = total_faiss_l2 = total_faiss_ivf = total_brute = 0.0

    # Option: use a generator to lazily yield the next image path
    # for img_path in iter_paths(img_paths_file):
    #     process(img_path)

    #for img_path in tqdm(paths, desc="Feature Extraction + Search in Gallery"):
    for img_path in tqdm(paths):
        start_feat = time.perf_counter()
        q_feat = extract_query_feature(img_path, model, preprocess, device)
        total_extract += time.perf_counter() - start_feat

        start_faiss_flat = time.perf_counter()
        _ = search_faiss(flat_l2_index, q_feat, args.k)
        total_faiss_l2 += time.perf_counter() - start_faiss_flat

        start_faiss_ivf = time.perf_counter()
        _ = search_faiss(ivf_index, q_feat, args.k)
        total_faiss_ivf += time.perf_counter() - start_faiss_ivf

        start_cos_brute = time.perf_counter()
        _ = brute_cosine_search(feats, q_feat, args.k)
        total_brute += time.perf_counter() - start_cos_brute

    n = len(paths)
    logging.info("Size of search gallery: %d", n)
    logging.info("Query extract: %.5fs", total_extract / n)
    logging.info("Brute cosine: %.5fs", total_brute / n)
    logging.info("Flat L2 search: %.5fs", total_faiss_l2 / n)
    logging.info("Flat IVF search: %.5fs", total_faiss_ivf / n)


if __name__ == "__main__":
    main()
