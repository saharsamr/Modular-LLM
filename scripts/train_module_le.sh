#!/bin/sh
python ../train_experts.py \
    --language_expert=1 \
    --language='en' \
    --le_train_json_path='/content/en_Wiki_10k_LM_511_1.json' \
    --le_test_json_path='/content/en_Wiki_10k_LM_511_1_test.json' \
    --cluster_idx=0 \
    --batch_size 2 \
    --seed 1234
