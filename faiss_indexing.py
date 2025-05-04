import argparse
import os
import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from feature_extractor import get_feature_extractor
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description='Extract features and build FAISS indexes'
    )
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing input images')
    parser.add_argument('--out-dir', required=True,
                        help='Directory to save features and indexes')
    parser.add_argument('--model', default='resnet18',
        choices=['resnet18', 'mobilenet', 'clip'],
        help='Backbone model for feature extraction (options: resnet18, mobilenet, clip)')
    parser.add_argument('--bs', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--metric', choices=['l2', 'ip'], default='ip',
                        help='Distance metric: l2 or inner-product')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feats_path = os.path.join(args.out_dir, 'feats.npy')
    imgs_path = os.path.join(args.out_dir, 'img_paths.txt')
    flat_idx_path = os.path.join(args.out_dir, 'flat.idx')
    ivf_idx_path = os.path.join(args.out_dir, 'ivf.idx')

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = get_feature_extractor(
        args.model, pretrained=True, device=device
    )
    model.eval()

    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Gather image paths
    image_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])

    #saving the image paths that are sorted to index later
    with open(imgs_path, "w") as f:
        f.write("\n".join(image_files))


    # Extract and normalize features
    feats_list = []
    for idx in tqdm(range(0, len(image_files), args.bs), desc="Processing images"):
        batch_files = image_files[idx: idx + args.bs]
        tensors = []
        for img_path in batch_files:
            img = Image.open(img_path).convert('RGB')
            tensors.append(preprocess(img).unsqueeze(0))

        batch_tensor = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            features = model(batch_tensor)
            normed = torch.nn.functional.normalize(features, p=2, dim=1).cpu().numpy()
        feats_list.append(normed)


    feats = np.vstack(feats_list).astype('float32')
    print("Number of Feature points and feature dimension -->",feats.shape)
    np.save(feats_path, feats)

    # -------------------------------------------------
    # Want to apply cosine similarity search
    # So we normalize feature vectors so that dot product = cos(θ):
    #   cos(θ) = (u/|u|) · (v/|v|)  ∈ [−1, 1]
    # From Cauchy-Schwarz:  |a·b| ≤ ‖a‖‖b‖
    # -------------------------------------------------

    # Build Flat index
    dim = feats.shape[1]
    if args.metric == 'ip':
        flat_index = faiss.IndexFlatIP(dim)
        ivf_metric = faiss.METRIC_INNER_PRODUCT
    else:
        flat_index = faiss.IndexFlatL2(dim)
        ivf_metric = faiss.METRIC_L2

    flat_index.add(feats)
    faiss.write_index(flat_index, flat_idx_path)

    # Build IVF index with same metric
    nlist = int(np.sqrt(len(feats)))
    quantizer = (faiss.IndexFlatIP(dim) if args.metric == 'ip' else faiss.IndexFlatL2(dim))
    ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, ivf_metric)
    ivf_index.train(feats)
    ivf_index.add(feats)
    faiss.write_index(ivf_index, ivf_idx_path)

    print(f"Saved features to {feats_path}")
    print(f"Saved sorted image paths to {imgs_path}")
    print(f"Built Flat index ({flat_index.ntotal} vectors)")
    print(f"Built IVF index ({ivf_index.ntotal} vectors, {nlist} clusters)")


if __name__ == '__main__':
    main()
