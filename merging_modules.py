from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import random
import numpy as np

from utils.arg_parser import experts_merging_arg_parser
from merging_lora_modules.simple_averaging import SimpleAveraging
from merging_lora_modules.xlora_average import XLoraAveraging
from merging_lora_modules.arrow_routing import ArrowRouting
from data_handler.dataset import (
    apply_preprocessing,
    create_message_column_for_test
)
from utils.metrics import compute_generation_metrics
from utils.config import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = experts_merging_arg_parser()
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, quantization_config=bnb_config)

    if args.merging_strategy == "simple_average":
        expert_merger = SimpleAveraging(model, tokenizer, args.model_name)
        expert_merger.merge()
        model = expert_merger.get_model()

    elif args.merging_strategy == 'xlora_average':

        if args.checkpoint_path:
            expert_merger = XLoraAveraging(model, tokenizer, args.model_name, args.data_portion)
            expert_merger.merge(load_path=args.checkpoint_path)
            model = expert_merger.get_model()
        else:
            expert_merger = XLoraAveraging(model, tokenizer, args.model_name, args.data_portion)
            expert_merger.merge()
            model = expert_merger.get_model()
    
    elif args.merging_strategy == 'arrow_routing':
        expert_merger = ArrowRouting(model, tokenizer, args.model_name)
        # vectors_dict, eigvals_dict = expert_merger.routing_function()
        model = expert_merger.merge(k=3)
        raise NotImplementedError

    elif args.merging_strategy == 'phi3':
        pass

    else:
        raise f'{args.merging_strategy} is not supported.'

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, truncation=True)

    routing_test_dataset = load_dataset("TahaBa/flan-routing-MoE-dataset", cache_dir="../data/")['test']
    routing_test_dataset = routing_test_dataset if args.data_portion == 1.0 \
        else routing_test_dataset.train_test_split(test_size=1-args.data_portion)['train']
    routing_test_dataset = routing_test_dataset.map(create_message_column_for_test)
    routing_test_dataset = routing_test_dataset.map(
        lambda sample:
        {'text': pipe.tokenizer.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=True)}
    )

    test_dataloader = DataLoader(routing_test_dataset, batch_size=args.batch_size)
    references, predictions = [], []
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        outputs = pipe(batch['text'], max_new_tokens=100)
        preds = [output[0]['generated_text'].split("<|assistant|>\n")[1].strip() for output in outputs]

        references.extend(batch['target'])
        predictions.extend(preds)

    metrics = compute_generation_metrics(references, predictions)
    print('=' * 100)
    print('BLEU:', metrics['bleu'])
    print('ROUGE:', metrics['rouge'])
    print('BERTSCORE:', metrics['bertscore'])
    print('=' * 100)
