#!/bin/sh
python ../train_experts.py \
    --cluster_idx=0 \
    --batch_size 32 \
    --seed 1234
