import argparse
import os
import time

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from feature_extractor import get_feature_extractor

from tqdm import tqdm

def extract_query_feat(path, model, preprocess, device):
    img = Image.open(path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1).cpu().numpy()
    return feat


def load_faiss_index(path):
    return faiss.read_index(path)

def search_faiss(index, q_feat, k):
    D, I = index.search(q_feat.astype('float32'), k)
    return D[0], I[0]


def brute_cosine_search(feats, q_feat, k):
    '''

    feats: num_gallery,feat_dim
    q_feat: feat_dim
    q_feat.T: (feat_dim,1)

    '''
    # (num_gallery,feat_dim) * (feat_dim,1) -> (num_gallery,1)
    sims = feats.dot(q_feat.T).squeeze()
    idx = np.argsort(sims)[-k:][::-1]
    return sims[idx], idx


def main():
    parser = argparse.ArgumentParser(description='Search query image against FAISS and brute-force searches')
    parser.add_argument('--out-dir',  required=True, help='Directory with saved indexes and features')
    parser.add_argument('--model',    default='resnet18', choices=['resnet18','mobilenet','clip'], help='Backbone model')
    parser.add_argument('--k',        type=int, default=5, help='Number of top results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = get_feature_extractor(args.model, pretrained=True, device=device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Load paths
    # doing this search experiment one by one for all images
    search_dir = os.path.join(args.out_dir, "img_paths.txt")
    with open(search_dir) as f:
        paths = [line.strip() for line in f]


    # loading the index and feature files

    # faiss flat
    flat_l2 = os.path.join(args.out_dir, "flat.idx")
    idx_l2 = load_faiss_index(flat_l2)

    # faiss ANN
    flat_ip = os.path.join(args.out_dir, "ivf.idx")
    idx_ip = load_faiss_index(flat_ip)

    # feature load
    feats_path = os.path.join(args.out_dir, "feats.npy")
    feats = np.load(feats_path)

    avg_extract =0
    avg_faiss_l2=0
    avg_faiss_cos=0
    avg_brute_cos = 0


    for img_paths in tqdm(paths):

        # Extract query feature
        tq_start = time.perf_counter()
        q_feat = extract_query_feat(img_paths, model, preprocess, device)
        tq = time.perf_counter() - tq_start
        avg_extract+=tq

        # FAISS L2 search
        t_faiss_l2_start = time.perf_counter()
        D_l2, I_l2 = search_faiss(idx_l2, q_feat, args.k)
        t_faiss_l2 = time.perf_counter() - t_faiss_l2_start
        avg_faiss_l2+=t_faiss_l2

        # FAISS IP search
        t_faiss_cos_start = time.perf_counter()
        D_ip, I_ip = search_faiss(idx_ip, q_feat, args.k)
        t_faiss_cos = time.perf_counter() - t_faiss_cos_start
        avg_faiss_cos+=t_faiss_cos

        # Brute cosine search
        t_brute_linear_start = time.perf_counter()
        D_cos, I_cos = brute_cosine_search(feats, q_feat, args.k)
        t_brute_linear = time.perf_counter() - t_brute_linear_start
        avg_brute_cos+=t_brute_linear


    # Print timings
    N=len(paths)
    print(f"Size of Search Gallery {N}")
    print(f"Query extract: {avg_extract/N:.5f}s")
    print(f"Brute cosine: {avg_brute_cos / N:.5f}s")
    print(f"Flat L2 search: {avg_faiss_l2/N:.5f}s")
    print(f"Flat IP search: {avg_faiss_cos/N:.5f}s")


if __name__ == '__main__':
    main()
