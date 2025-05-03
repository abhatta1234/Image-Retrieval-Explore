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

# ———————————————
# 1) CONFIG
# ———————————————
DATA_DIR = "/scratch365/abhatta/search_webapp/search_gallery"       # your image directory
QUERY_DIR = "/path/to/queries"      # images you want to search with
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 32

# output files
FEATURES_FILE   = "features.npy"
FLAT_INDEX_FILE = "flatl2.index"
IVF_INDEX_FILE  = "ivf.index"
IMAGE_PATHS     = "image_paths.txt"

# ———————————————
# 2) SET UP MODEL (ResNet-18 → 512-dim)
# ———————————————
model = models.resnet18(pretrained=True)
# just make the feat-dim to class num mapping to just remain feat-dim
model.fc = nn.Identity()
model = model.to(DEVICE).eval()

preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


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
            batch_features = model(batch_tensor).cpu().numpy()

        all_features.append(batch_features)
        all_paths.extend(valid_paths)

    if all_features:
        return np.vstack(all_features), all_paths
    else:
        return np.array([]), []

# ———————————————
# 3) BUILD & SAVE GALLERY INDEXES
# ———————————————
# (a) extract gallery features
gallery_feats, gallery_paths = extract_features(DATA_DIR)

# (b) save raw features if you want
np.save(FEATURES_FILE, gallery_feats)
with open(IMAGE_PATHS, "w") as f:
    f.write("\n".join(gallery_paths))

# (c) Flat L2 index
d = gallery_feats.shape[1]
flat_idx = faiss.IndexFlatL2(d)
flat_idx.add(gallery_feats)
faiss.write_index(flat_idx, FLAT_INDEX_FILE)

# (d) IVF index (choose #clusters = sqrt(N) as heuristic)
nlist = int(np.sqrt(len(gallery_feats)))
quantizer = faiss.IndexFlatL2(d)
ivf_idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
ivf_idx.train(gallery_feats)
ivf_idx.add(gallery_feats)
faiss.write_index(ivf_idx, IVF_INDEX_FILE)

print("Saved:")
print(f" • {FEATURES_FILE} ({gallery_feats.shape})")
print(f" • {FLAT_INDEX_FILE}  (ntotal={flat_idx.ntotal})")
print(f" • {IVF_INDEX_FILE}   (ntotal={ivf_idx.ntotal}, nlist={nlist})")

