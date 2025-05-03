import faiss
import numpy as np
from PIL import Image
from torchvision import models, transforms, datasets
import torch
from torch import nn
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import time

# 1) LOAD INDEX + PATHS
index = faiss.read_index("/scratch365/abhatta/search_webapp/flatl2.index")         # or ivf.index
with open("/scratch365/abhatta/search_webapp/image_paths.txt") as f:
    gallery_paths = [line.strip() for line in f]


def extract_image_feat(image_path: str) -> np.ndarray:
    """
    Load one image, preprocess, and extract a 1×512 float32 feature vector.
    """
    img = Image.open(image_path).convert("RGB")
    x   = preproc(img).unsqueeze(0).to(device)    # shape: [1, 3, 224, 224]
    with torch.no_grad():
        feat = model(x).cpu().numpy()             # shape: [1, 512]
    return feat.astype('float32')                 # ensure faiss compatibility


# 2) SET UP MODEL (same as before)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)
model.fc = nn.Identity()
model.to(device).eval()

preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


query_feat = extract_image_feat("/scratch365/abhatta/search_webapp/tiny-imagenet-200/test/images/test_0.JPEG")
# start timer
start = time.perf_counter()
D, I = index.search(query_feat, k=5)
elapsed = time.perf_counter() - start

# # for index in I[0]:
# #     print(index)
# #     shutil.copy(gallery_paths[index],"/scratch365/abhatta/search_webapp/test-checks-ivf")
#
# print(query_feat.shape)
#
#
#
# all_features = np.load("/scratch365/abhatta/search_webapp/features.npy")
# # Compute cosine similarities
# similarities = cosine_similarity(query_feat, all_features)  # shape: (1, 100000)
#
# # Get top 5 indices and values
# top5_indices = np.argsort(similarities[0])[-5:][::-1]
#
#
# top5_similarities = similarities[0, top5_indices]
#
# print("Top 5 indices:", top5_indices)
# print("Top 5 similarities:", top5_similarities)
print(f"Elapsed time: {elapsed:.4f} s")

