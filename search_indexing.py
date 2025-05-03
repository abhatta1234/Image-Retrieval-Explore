
import argparse
import os
import time

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from feature_extractor import get_feature_extractor


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
    sims = feats.dot(q_feat.T).squeeze()
    idx = np.argsort(sims)[-k:][::-1]
    return sims[idx], idx


def main():
    parser = argparse.ArgumentParser(
        description='Search query image against FAISS and brute-force searches'
    )
    parser.add_argument('--query',    required=True, help='Query image path')
    parser.add_argument('--out-dir',  required=True, help='Directory with saved indexes and features')
    parser.add_argument('--model',    default='resnet18', choices=['resnet18','mobilenet','clip'], help='Backbone model')
    parser.add_argument('--flat-l2',  required=True, help='Flat L2 FAISS index file')
    parser.add_argument('--flat-ip',  required=True, help='Flat IP FAISS index file')
    parser.add_argument('--features', required=True, help='NumPy file of gallery features')
    parser.add_argument('--paths',    required=True, help='Text file of gallery image paths')
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

    # Extract query feature
    t0 = time.perf_counter()
    q_feat = extract_query_feat(args.query, model, preprocess, device)
    tq = time.perf_counter() - t0

    # FAISS L2 search
    idx_l2 = load_faiss_index(args.flat_l2)
    t0 = time.perf_counter()
    D_l2, I_l2 = search_faiss(idx_l2, q_feat, args.k)
    t_l2 = time.perf_counter() - t0

    # FAISS IP search
    idx_ip = load_faiss_index(args.flat_ip)
    t0 = time.perf_counter()
    D_ip, I_ip = search_faiss(idx_ip, q_feat, args.k)
    t_ip = time.perf_counter() - t0

    # Brute cosine search
    feats = np.load(args.features)
    t0 = time.perf_counter()
    D_cos, I_cos = brute_cosine_search(feats, q_feat, args.k)
    t_cos = time.perf_counter() - t0

    # Load paths
    with open(args.paths) as f:
        paths = [line.strip() for line in f]

    # Print timings
    print(f"Query extract: {tq:.4f}s")
    print(f"Flat L2 search: {t_l2:.4f}s")
    print(f"Flat IP search: {t_ip:.4f}s")
    print(f"Brute cosine: {t_cos:.4f}s")

    # Print top-k results
    print("\nTop-k results:")
    def display(name, I, D):
        print(f"\n{name}:")
        for idx, score in zip(I, D):
            print(f"  {paths[idx]} (score={score:.4f})")
    display('L2', I_l2, D_l2)
    display('IP', I_ip, D_ip)
    display('Cosine', I_cos, D_cos)

if __name__ == '__main__':
    main()
