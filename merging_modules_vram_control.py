from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import random
import numpy as np

from utils.arg_parser import experts_merging_arg_parser
from merging_lora_modules.simple_averaging import SimpleAveraging
from merging_lora_modules.xlora_average import XLoraAveraging
from merging_lora_modules.arrow_routing import ArrowRouting
from data_handler.test_datasets import (
    read_test_dataset,
    create_zero_shot_message,
    map_output_to_desired_target,
    create_multi_choice_options,
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
    
    pipe = pipeline(task="text-generation", model=strategy_model, tokenizer=tokenizer, truncation=True, padding=True)

    routing_test_dataset = read_test_dataset(args.dataset_name)
    routing_test_dataset = routing_test_dataset if args.data_portion == 1.0 \
        else routing_test_dataset.train_test_split(test_size=1-args.data_portion)['train']

    if args.test_type == 'zero_shot':
        routing_test_dataset = routing_test_dataset.map(create_multi_choice_options,
                                                        fn_kwargs={'ds_name': args.dataset_name})

    routing_test_dataset = routing_test_dataset.map(
        lambda sample:
        {
            'Options': {
                'text': pipe.tokenizer.apply_chat_template(
                    option['messages'], tokenize=True, add_generation_prompt=False)
            } for option in sample['options']
        }
    )

    print(routing_test_dataset[0])
    exit(0)

    # if args.test_type == 'zero_shot':
    #     routing_test_dataset = routing_test_dataset.map(create_zero_shot_message, fn_kwargs={'ds_name':args.dataset_name})
    # elif args.test_type == 'few_shot':
    #     pass
    #     # routing_test_dataset = routing_test_dataset.map(create_few_shot_message)
    #
    # routing_test_dataset = routing_test_dataset.map(
    #     lambda sample:
    #     {'text': pipe.tokenizer.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=True)}
    # )
    #
    # test_dataloader = DataLoader(routing_test_dataset, batch_size=args.batch_size)
    # inputs, references, predictions = [], [], []
    # for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    #     # Calling the model's forward path to apply Arrow Routing
    #
    #     tokenized_batch = tokenizer(batch['text'], return_tensors="pt", truncation=True, padding=True).to('cuda')
    #     if len(tokenized_batch['input_ids'][0]) > 1500:
    #         continue
    #
    #     if args.merging_strategy == 'arrow_routing':
    #         strategy_model(**tokenized_batch, compute_arrow_weights=True, top_k=3)
    #     elif args.merging_strategy == 'phi3':
    #         strategy_model(**tokenized_batch)
    #
    #     # Generate the answer using the new adapter
    #     outputs = pipe(batch['text'], max_new_tokens=100)
    #     preds = [output[0]['generated_text'].split("<|assistant|>\n")[1].strip() for output in outputs]
    #
    #     references.extend([map_output_to_desired_target(args.dataset_name, batch)])
    #     predictions.extend(preds)
    #     inputs.extend(batch['text'])
    #     print(batch['text'])
    #     print(references[-1])
    #     print(preds)
    #     print('==============')
    #
    # metrics = compute_generation_metrics(references, predictions)
    # print('=' * 100)
    # print('BLEU:', metrics['bleu'])
    # print('ROUGE:', metrics['rouge'])
    # print('BERTSCORE:', metrics['bertscore'])
    # print('=' * 100)
