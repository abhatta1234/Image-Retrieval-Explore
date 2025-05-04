#!/bin/bash
# Grid engine directives
#$ -o ./logs
#$ -pe smp 8
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3
#$ -N faiss_indexing

# Define models array
model=(resnet18 mobilenet clip)

# Activate conda environment
conda activate search

# Run indexing script with appropriate parameters
python ./faiss_indexing.py \
  --data-dir ./search_gallery \
  --out-dir ./output/${model[${SGE_TASK_ID}-1]} \
  --model ${model[${SGE_TASK_ID}-1]} \
  --bs 32