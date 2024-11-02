import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from sklearn.metrics import accuracy_score

from data_handler.test_datasets import (
    read_test_dataset,
    create_multi_choice_options,
    extract_multi_choice_target_index,
)
from merging_lora_modules.arrow_routing import ArrowRouting
from merging_lora_modules.simple_averaging import SimpleAveraging
from merging_lora_modules.xlora_average import XLoraAveraging
from utils.arg_parser import experts_merging_arg_parser
from utils.config import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loglike_loss(logits, labels, reduction="none"):
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # reshape back
    if reduction == "none":
        loss = loss.view((bs, -1))
        # mean only non-zero
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss
    return loss


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
        strategy_model = expert_merger.get_model()

    elif args.merging_strategy == 'xlora_average':

        if args.checkpoint_path:
            expert_merger = XLoraAveraging(model, tokenizer, args.model_name, args.data_portion)
            expert_merger.merge(load_path=args.checkpoint_path)
            strategy_model = expert_merger.get_model()
        else:
            expert_merger = XLoraAveraging(model, tokenizer, args.model_name, args.data_portion)
            expert_merger.merge()
            strategy_model = expert_merger.get_model()

    elif args.merging_strategy == 'arrow_routing':
        # ًWe only load the model with all the adapters here, the merging will be done inside the model's layer
        expert_merger = ArrowRouting(model, tokenizer, args.model_name)
        strategy_model = expert_merger.get_model()

    elif args.merging_strategy == 'phi3':
        expert_merger = None
        strategy_model = model

    else:
        raise f'{args.merging_strategy} is not supported.'

    routing_test_dataset = read_test_dataset(args.dataset_name)
    routing_test_dataset = routing_test_dataset if args.data_portion == 1.0 \
        else routing_test_dataset.train_test_split(test_size=1 - args.data_portion)['train']

    labels, predictions = [], []
    with torch.no_grad():
        for i, sample in tqdm(enumerate(routing_test_dataset)):
            options = create_multi_choice_options(sample, args.dataset_name)
            option_losses = []
            for option in options:
                tokenized_text = tokenizer(
                    text=option, text_target=option, return_tensors='pt', truncation=True, max_length=512).to('cuda')
                if args.merging_strategy == 'arrow_routing':
                    logits = strategy_model(tokenized_text['input_ids'], compute_arrow_weights=True, top_k=3).logits
                else:
                    logits = strategy_model(tokenized_text['input_ids']).logits
                loss = compute_loglike_loss(logits, tokenized_text['labels'])
                option_losses.append(loss.to('cpu'))

            labels.append(extract_multi_choice_target_index(sample, args.dataset_name))
            predictions.append(np.argmin(option_losses))

        print(f'Accuracy for dataset {args.dataset_name} and strategy {args.merging_strategy} is: '
              f'{accuracy_score(labels, predictions)}')
