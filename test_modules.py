from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

import torch
import numpy as np
import random

from utils.arg_parser import test_arg_parser
from data_handler.dataset import read_dataset, create_message_column_for_test
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, padding_side='right', model_max_length=MAX_LENGTH
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.model_checkpoint_path).to("cuda")
    model = model.merge_and_unload()
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    test_data = read_dataset(args.dataset_name, args.cluster_idx, args.data_portion, return_test=True)
    test_data = test_data.map(create_message_column_for_test)
    test_data = [
        pipe.tokeizer.apply_chat_template(
            sample['messages'], tokenize=False, add_generation_prompt=True)
        for sample in test_data]

    outputs = pipe(
        test_data, batch_size=args.batch_size, max_new_tokens=100,
        do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95
    )
    preds = [outputs[i][0]['generated_text'].split("<|assistant|>\n")[1].strip() for i in range(len(outputs))]
    references = [test_data[i]['target'] for i in range(len(outputs))]

    metrics = compute_experts_metrics(references, preds)
    print('=' * 100)
    print('BLEU:', metrics['bleu'])
    print('ROUGE:', metrics['rouge'])
    print('BERTSCORE:', metrics['bertscore'])
    print('=' * 100)
