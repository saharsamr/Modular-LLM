from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from tqdm import tqdm

from utils.arg_parser import test_arg_parser
from data_handler.dataset import read_dataset
from utils.metrics import compute_experts_metrics
from utils.config import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = test_arg_parser()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint_path, use_fast=True)
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # TODO: what dtype?
            bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint_path,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto"
    )

    test_ds = read_dataset(args.dataset_name, args.cluster_idx, args.data_portion, return_test=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size)

    with torch.no_grad():
        for sample in test_ds:

            input_ids = tokenizer(
                sample['source'], padding='max_length', truncation=True,
                return_tensors='pt', max_length=MAX_SOURCE_TOKENS).input_ids.to("cuda")
            outputs = model.generate(
                    input_ids=input_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=100,
            )
            labels = sample['target']
            outputs = tokenizer.decode(outputs[len(input_ids):], skip_special_tokens=True)
            print(labels)
            print(outputs)
            print('--------------------------------')

            print(compute_experts_metrics(outputs, outputs))
