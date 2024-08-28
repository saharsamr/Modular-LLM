from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch


from utils.arg_parser import experts_merging_arg_parser
from merging_lora_modules.simple_averaging import SimpleAveraging
from merging_lora_modules.xlora_average import XLoraAveraging
from utils.config import *


if __name__ == "__main__":
    args = experts_merging_arg_parser()

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
    elif args.mering_strategy == 'xlora_average':
        expert_merger = XLoraAveraging(model, tokenizer, args.model_name)
    else:
        raise f'{args.merging_strategy} is not supported.'

    expert_merger.merge()
    model = expert_merger.get_model()
