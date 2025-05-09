# Image Retrieval Exploration: Optimizing Similarity Search with FAISS

The project begins by benchmarking three similarity‑search strategies on a **100 k**‑image subset of **tiny‑ImageNet**, quantifying their speed and memory footprints. It then scales the experiment to **1 M** images to evaluate two critical questions: **(i)** how much, if any, retrieval‑accuracy degradation arises when approximate‑nearest‑neighbor indexing replaces brute‑force search, and **(ii)** how the end‑to‑end latency profile shifts as the gallery size increases by an order of magnitude.

---

## Project Overview

The project compares **three** search approaches:

1. **Brute‑force cosine‑similarity search**
2. **FAISS flat cosine‑similarity search**
3. **Approximate Nearest Neighbor (ANN) search using FAISS**

### Feature‑extraction models

| Model                          | Parameter Count (Millions) |
| ------------------------------ | -------------------------- |
| ResNet‑18                      | \~11.7 M                   |
| MobileNet                      | \~4.2 M                    |
| CLIP Vision Encoder (ViT‑B/32) | \~87.85 M                  |

---

## Implementation Details

### Feature Extraction

All training images (100 k) from tiny‑ImageNet are processed to extract feature vectors with each model. These vectors are **normalized to unit length** so cosine similarity equals the dot product.

### Vector Normalization

```python
# Normalize feature vectors so that dot product = cos(θ)
#   cos(θ) = (u / |u|) · (v / |v|)  ∈ [−1, 1]
torch.nn.functional.normalize(features, p=2, dim=1)
```

* After normalization, cosine similarity ⇔ inner product.
* For unit vectors `|x−y|² = 2 − 2⟨x, y⟩`.

Further, for unit vectors

$$
|x - y|^{2} = |x|^{2} + |y|^{2} - 2\langle x, y\rangle
           = 2 - 2\cos(\theta)
$$

which enables conversion between distance and similarity metrics.

### FAISS Clustering (IVF)

For the ANN implementation, **`nlist`** (clusters) is key:

* Rule of thumb for 100 k vectors: `√N ≈ 316` to `N/10 = 10 000`
* Common practical values: 100, 256, 512, 1024
* Initial implementation uses **√N clusters**.

---

## Performance Results

Each of the 100 k images is:

1. Fed through each network for feature extraction
2. Queried against the gallery with each search method

Feature extraction uses an **NVIDIA Titan XP GPU**; search runs on CPU/compute nodes.

| Method                 | ResNet‑18 | MobileNet |    CLIP   |
| :--------------------- | :-------: | :-------: | :-------: |
| **Feature dim**        |    512    |    1280   |    512    |
| **Feature extraction** | 0.03686 s | 0.02060 s | 0.03746 s |
| **Brute‑Search**       | 0.02646 s | 0.04427 s | 0.02701 s |
| **FAISS‑Flat**         | 0.01635 s | 0.03949 s | 0.01757 s |
| **FAISS‑IVF**          | 0.00016 s | 0.00035 s | 0.00017 s |

> *MobileNet’s larger feature dimension explains its higher search time, while its inference time is the lowest due to low MFLOPS.
> Feature extraction (per image) dominates latency; network inference overshadows data‑transfer costs.*

---

## Key Insights

Feature‑extraction time, **not** search time, dominates end‑to‑end latency. Efficient libraries such as **FAISS** make even large‑scale search (\~100 k items) extremely fast, underscoring the need to optimize the extraction phase.

---

## Accuracy vs. Trade‑off Plot

*To be added.*

---

## Feature‑Extraction & FAISS‑Indexing Commands

### Extract features & create FAISS index files

```bash
python faiss_indexing.py \
  --data-dir <dir_where_all_images_are> \
  --out-dir <directory_save_index_features> \
  --model <resnet|clip|mobilenet> \
  --bs <batch_size>
```

### Reproduce **Table 1** results

```bash
python3 search_index_timing.py \
  --model <resnet|clip|mobilenet>
```

### Generate accuracy vs. trade‑off plot

(will be uploaded soon)

```bash
python3 search_vs_accuracy_tradeoff.py \
  --all_img_list <path_to_txt_with_image_paths> \
  --model <resnet|clip|mobilenet> \
  --metric <l2|ip> \
  --k <top_k_retrieval>
```

---

## To‑Dos

* [ ] Deploy a smaller end‑to‑end web demo for development practice.

