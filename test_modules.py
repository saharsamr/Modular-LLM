from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
from torch.utils.data import DataLoader
import torch
import numpy as np
import gc

import random
from tqdm import tqdm

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
    print('Loading Model ...')
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint_path, torch_dtype=torch.float16, quantization_config=bnb_config)
    model = PeftModel.from_pretrained(model, args.model_checkpoint_path).to("cuda")
    model = model.merge_and_unload()

    print('Initializing Pipeline ...')
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, truncation=True)

    print('Loading and Processing Data ...')
    test_data = read_dataset(args.dataset_name, args.cluster_idx, args.data_portion, return_test=True, seed=args.seed)
    test_data = test_data.map(create_message_column_for_test)
    test_data = test_data.map(
        lambda sample:
        {'text': pipe.tokenizer.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=True)}
    )
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    torch.cuda.empty_cache()
    gc.collect()
    
    print('Generating Predictions ...')
    references = []
    predictions = []
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        outputs = pipe(batch['text'], max_new_tokens=100)
        preds = [output[0]['generated_text'].split("<|assistant|>\n")[1].strip() for output in outputs]

        references.extend(batch['target'])
        predictions.extend(preds)

        if (i % 100) == 0:
            torch.cuda.empty_cache()
            gc.collect()

    metrics = compute_experts_metrics(references, predictions)
    print('=' * 100)
    print('BLEU:', metrics['bleu'])
    print('ROUGE:', metrics['rouge'])
    print('BERTSCORE:', metrics['bertscore'])
    print('=' * 100)
