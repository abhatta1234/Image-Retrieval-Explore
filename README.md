# Image Retrieval Exploration: Optimizing Similarity Search with FAISS

This project explores performance differences in similarity search methods using FAISS index for image retrieval across a dataset of 100k images from tiny-imagenet.

## Project Overview

The project compares three different search approaches:
1. Brute force cosine similarity search
2. FAISS flat cosine similarity search
3. Approximate Nearest Neighbor (ANN) search using FAISS

Features are extracted using three different models:

| Model                        | Parameter Count (Millions) |
|------------------------------|---------------------------|
| ResNet18                     | ~11.7M                    |
| MobileNet                    | ~4.2M                     |
| CLIP Vision Encoder (ViT-B/32) | ~87.85M                 |


## Implementation Details

### Feature Extraction

All training images (100k) from tiny-imagenet are processed to extract feature vectors using each model. These feature vectors are then normalized to unit length to enable cosine similarity calculation through dot products.

### Vector Normalization

For cosine similarity search:
```
# Want to apply cosine similarity search
# So we normalize feature vectors so that dot product = cos(θ):
#   cos(θ) = (u/|u|) · (v/|v|)  ∈ [−1, 1]
# From Cauchy-Schwarz:  |a·b| ≤ ‖a‖‖b‖
```

Implementation details:
- Vectors are normalized during feature extraction and inference: `torch.nn.functional.normalize(features, p=2, dim=1)`
- After normalization, cosine similarity is equivalent to inner product
- For normalized vectors: |x−y|² = 2−2×⟨x,y⟩


- The relationship between L2 distance and cosine similarity for normalized vectors:
  - |x−y|² = |x|² + |y|² - 2⟨x,y⟩
  - For unit vectors where |x| = |y| = 1:
  - |x−y|² = 2 - 2⟨x,y⟩ = 2 - 2cos(θ)
This allows converting between distance and similarity metrics

    
### FAISS Clustering

For the ANN implementation, the number of clusters (nlist) is a key parameter:
- Rule of thumb: between √N (≈316) and N/10 (10,000) for 100k vectors
- Common practical values: 100, 256, 512, or 1024
- Initial implementation uses √N clusters

## Performance Results

To calculate average timing, each of the 100K images was processed through each neural network for feature extraction and then searched using each method. The reported values represent the average of these 100K runs. Feature extraction was performed using an NVIDIA Titan XP GPU, while search operations were conducted on CPU/compute nodes.

| Method             | ResNet18  | Mobilenet   |   CLIP    |
|:-------------------|:---------:|:-----------:|:---------:|
| Feature dim        |    512    |    1280     |    512    |
| Feature extraction |  0.03686s |   0.02060s  |  0.03746s |
| Brute-Search       |  0.02646s |   0.04427s  |  0.02701s |
| FAISS-Flat         |  0.01635s |   0.03949s  |  0.01757s |
| FAISS-IVF          |  0.00016s |   0.00035s  |  0.00017s |


> Search time for MobileNet is higher due to its larger feature dimension.  
> MobileNet's inference time is the lowest, as it has low MFLOPS.  
> Note: Feature extraction is performed per single image, so network inference time dominates rather than data transfer costs.

## Key Insights

Feature extraction time dominates the overall latency while search operations are surprisingly fast, even over large galleries, when using efficient libraries like FAISS. This was an unexpected finding that highlights the importance of optimization in the feature extraction phase.

## Usage

```
python ./faiss_indexing.py \
  --data-dir ./search_gallery \
  --out-dir ./output/${model[${SGE_TASK_ID}-1]} \
  --model ${model[${SGE_TASK_ID}-1]} \
  --bs 32

```

## Future Work

1. Extend to larger search sets (full ImageNet)
2. Conduct a comprehensive ANN accuracy vs. speed study (effects may be more pronounced with larger datasets)
3. Deploy a smaller version on web as an end-to-end demo for web development practice
