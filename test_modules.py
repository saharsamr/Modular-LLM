from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

import torch
import numpy as np
import random

from utils.arg_parser import test_arg_parser
from data_handler.dataset import read_dataset
from utils.metrics import compute_experts_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = test_arg_parser()
    set_seed(args.seed)

    test_ds = read_dataset(args.dataset_name, args.cluster_idx, args.data_portion, return_test=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # TODO: what dtype?
            bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # Test with one sample
    idx = 70
    input_ids = tokenizer(test_ds['source'][idx], return_tensors="pt").input_ids.to("cuda")

    # inference
    outputs = model.generate(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
            do_sample=True,
            repetition_penalty=2.0,
            temperature=0.7,
            top_k=3,
            top_p=0.95
    )
    labels = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    labels = [label.split('### Response:')[-1] for label in labels]
    outputs = [output.split('### Response:')[-1] for output in outputs]

    print(compute_experts_metrics(labels, outputs))
    # print(test_ds['source'][idx])
    # print('-'*100)
    # print('-'*100)
    # print(test_ds['target'][idx])
    # rouge = evaluate.load('rouge')
    # results = rouge.compute(predictions=[generated_text], references=[reference])
    # print(results)

    

    
