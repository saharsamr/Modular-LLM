#!/bin/sh
for cluster_idx in $(seq 0 9); do
    python ../train_experts.py \
        --cluster_idx=$cluster_idx \
        --batch_size 1 \
        --seed 1234
done