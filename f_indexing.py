import os
import torch
import numpy as np
import faiss
from torchvision import models, transforms, datasets
from torch import nn
from tqdm import tqdm


import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
from tqdm import tqdm
from feature_extractor import get_feature_extractor

# ———————————————
# 1) CONFIG
# ———————————————
DATA_DIR = "/scratch365/abhatta/search_webapp/search_gallery"       # your image directory
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 32

# output files
# "resnet18"  # or "mobilenet" or "clip"
model_name = "resnet18"
OUTPUT_DIR = f"/scratch365/abhatta/search_webapp/{model_name}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_FILE   = os.path.join(OUTPUT_DIR,"features.npy")
FLAT_INDEX_FILE = os.path.join(OUTPUT_DIR,"flatl2.index")
IVF_INDEX_FILE  = os.path.join(OUTPUT_DIR,"ivf.index")
IMAGE_PATHS     = os.path.join(OUTPUT_DIR,"image_paths.txt")

# ———————————————
# 2) SET UP MODEL (ResNet-18 → 512-dim)
# ———————————————
model, feature_dim = get_feature_extractor(model_name, pretrained=True, device=DEVICE)

preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ———————————————
# 3) Manually batchify and extract features
# Note: if the data dir follow standard class dir structure - ImageFolder can be used directly
# Just wanted to try batching manually
# ———————————————

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'))


def extract_features(data_dir):
    """Extract features from all images in the data directory"""
    # Get all image paths directly without using DatasetFolder
    image_paths = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            image_paths.append(os.path.join(data_dir, file))

    print(f"Found {len(image_paths)} images")

    # Process in batches
    all_features = []
    all_paths = []

    for i in tqdm(range(0, len(image_paths), BATCH)):
        batch_paths = image_paths[i:i + BATCH]
        batch_images = []
        valid_paths = []

        # Process each image
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = preproc(img).unsqueeze(0)
                batch_images.append(img_tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if not batch_images:
            continue

        # Stack images into a batch
        batch_tensor = torch.cat(batch_images).to(DEVICE)

        # Extract features
        with torch.no_grad():
            batch_features = model(batch_tensor)
            norm_features = torch.nn.functional.normalize(batch_features,p=2,dim=1).cpu().numpy()
        all_features.append(norm_features)
        all_paths.extend(valid_paths)

    if all_features:
        return np.vstack(all_features), all_paths
    else:
        return np.array([]), []

# ———————————————
# 4) BUILD & SAVE GALLERY INDEXES
# Want to apply cosine similarity search
# So saving the normalize feature vector to make dot product as cosine similarity metric
# cos(θ) = (u/|u|) . (v/|v|) --> Bounded betweeen cosθ ∈ [−1,1]
# From Cauchy-Schwarz: The absolute value of an inner (dot) product can never exceed the product of the two vectors’ lengths.
# -- ∣a⋅b∣≤∥a∥∥b∥.
# ———————————————

# (a) extract gallery features

gallery_feats, gallery_paths = extract_features(DATA_DIR)

# (b) save raw features if you want
np.save(FEATURES_FILE, gallery_feats)
with open(IMAGE_PATHS, "w") as f:
    f.write("\n".join(gallery_paths))

# (c) Flat L2 index
d = gallery_feats.shape[1]
flat_cos_idx = faiss.IndexFlatIP(d)
# np defaults to float64 and this might cause error for faiss
# so doing float32 conversion
flat_cos_idx.add(gallery_feats.astype('float32'))
faiss.write_index(flat_cos_idx, FLAT_INDEX_FILE)

# (d) IVF index with cosine-similarity (choose #clusters = sqrt(N) as heuristic)
# this can be done using l2-distance as well
nlist = int(np.sqrt(len(gallery_feats)))
quantizer = faiss.IndexFlatIP(d)
ivf_cos_idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
ivf_cos_idx.train(gallery_feats.astype('float32'))
ivf_cos_idx.add(gallery_feats.astype('float32'))
faiss.write_index(ivf_cos_idx, IVF_INDEX_FILE)

print("Saved:")
print(f" • {FEATURES_FILE} ({gallery_feats.shape})")
print(f" • {FLAT_INDEX_FILE}  (ntotal={flat_idx.ntotal})")
print(f" • {IVF_INDEX_FILE}   (ntotal={ivf_idx.ntotal}, nlist={nlist})")




