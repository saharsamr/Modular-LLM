import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score

from data_handler.test_datasets import (
    read_test_dataset,
    create_multi_choice_options,
    extract_multi_choice_target_index,
    split_dataset_by_option_count
)
from merging_lora_modules.arrow_routing import ArrowRouting
from merging_lora_modules.simple_averaging import SimpleAveraging
from merging_lora_modules.xlora_average import XLoraAveraging
from utils.arg_parser import experts_merging_arg_parser
from utils.config import *
from merging_lora_modules.cross_lingual_expert_organiser import CrossLingualExpertOrganiser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loglike_loss(logits, labels, pad_index, reduction="none"):
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=pad_index)
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


multi_choice_datasets = ['piqa', 'boolq', 'swag', 'hswag', 'arc-easy', 'arc-challenge', 'wg', 'oqa', 'bbh']


def evaluate_on_multi_choice(eval_dataloader_list, model, tokenizer, ds_name, routing_strategy):
    predictions, labels = [], []
    for data_loader in eval_dataloader_list:
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_options = create_multi_choice_options(batch, ds_name, tokenizer)
            batch_option_count = [len(sample) for sample in batch_options]
            batch_options = [option for sample in batch_options for option in sample]
            tokenized_text = tokenizer(
                    text=batch_options, text_target=batch_options, padding=True, return_tensors='pt', truncation=True, max_length=512).to('cuda')
            if routing_strategy == 'arrow_routing':
                logits = model(tokenized_text['input_ids'], tokenized_text['attention_mask'], compute_arrow_weights=True, top_k=3).logits
            else:
                logits = model(tokenized_text['input_ids'], tokenized_text['attention_mask']).logits
            loss = compute_loglike_loss(logits, tokenized_text['labels'], pad_index=tokenizer.pad_token_id).to('cpu')

            start = 0
            for option_count in batch_option_count:
                predictions.append(int(np.argmin(loss[start:start+option_count])))
                start += option_count
            labels.extend(extract_multi_choice_target_index(batch, args.dataset_name))

    print(f'Accuracy for dataset {args.dataset_name} and strategy {args.merging_strategy} is: '
          f'{accuracy_score(labels, predictions)}')


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

    if args.posthoc_cross_lingual or args.factorize_average_lora:
        cle_org = CrossLingualExpertOrganiser(
            model, tokenizer, args.model_name,
            args.source_formal_expert_path, args.target_formal_expert_path,
            args.disentanglement_method
        )
        cle_org.merge(args.add_functional_only, use_avg_lora=args.factorize_average_lora)
        model = cle_org.get_model()

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
        expert_merger = ArrowRouting(
            model, tokenizer, args.model_name, load_lora_modules=(not args.posthoc_cross_lingual))
        strategy_model = expert_merger.get_model()

    elif args.merging_strategy == 'phi3':
        expert_merger = None
        strategy_model = model

    else:
        raise f'{args.merging_strategy} is not supported.'

    strategy_model.eval()

    routing_test_dataset = read_test_dataset(args.dataset_name)
    # routing_test_dataset = routing_test_dataset.train_test_split(test_size=400, seed=args.seed)['test']
    dataset_list = split_dataset_by_option_count(routing_test_dataset, args.dataset_name)
    data_loader_list = [torch.utils.data.DataLoader(ds, batch_size=args.batch_size) for ds in dataset_list]

    labels, predictions = [], []
    with torch.no_grad():
        if args.dataset_name in multi_choice_datasets:
            evaluate_on_multi_choice(
                data_loader_list, strategy_model, tokenizer, args.dataset_name, args.merging_strategy
            )
