# Image Retrieval Exploration: Optimizing Similarity Search with FAISS

This project explores performance differences in similarity search methods using FAISS index for image retrieval across a dataset of 100k images from tiny-imagenet.

## Project Overview

The project compares three different search approaches:
1. Brute force cosine similarity search
2. FAISS flat cosine similarity search
3. Approximate Nearest Neighbor (ANN) search using FAISS

Features are extracted using three different models:
- ResNet18
- MobileNet
- CLIP Vision Encoder

## Implementation Details

### Feature Extraction

All training images (100k) from tiny-imagenet are processed to extract feature vectors using each model. These feature vectors are then normalized to unit length to enable cosine similarity calculation through dot products.

### Vector Normalization

For cosine similarity search:
- Vectors are normalized using `faiss.normalize_L2`
- After normalization, cosine similarity is equivalent to inner product
- For normalized vectors: |x−y|² = 2−2×⟨x,y⟩

### FAISS Clustering

For the ANN implementation, the number of clusters (nlist) is a key parameter:
- Rule of thumb: between √N (≈316) and N/10 (10,000) for 100k vectors
- Common practical values: 100, 256, 512, or 1024
- Initial implementation uses 256 clusters

## Performance Results

The performance comparison across models and search methods:

| Method | MobileNet | ResNet18 | CLIP |
|--------|-----------|----------|------|
| Feature extraction | -         | -        | -    |
| Brute-Search | -         | -        | -    |
| FAISS-Flat | -         | -        | -    |
| FAISS-IVF | -         | -        | -    |

> Note: Feature extraction is performed per single image, so network inference time dominates rather than data transfer costs.

## Key Insights

Feature extraction time dominates the overall latency while search operations are surprisingly fast, even over large galleries, when using efficient libraries like FAISS. This was an unexpected finding that highlights the importance of optimization in the feature extraction phase.

## Future Work

1. Extend to larger search sets (full ImageNet)
2. Conduct a comprehensive ANN accuracy vs. speed study (effects may be more pronounced with larger datasets)
3. Deploy a smaller version on web as an end-to-end demo for web development practice