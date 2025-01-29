import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    LoftQConfig,
    PeftModel
)
from trl import SFTTrainer

from data_handler.dataset import (
    create_message_column,
    apply_preprocessing
)
from utils.config import *


class ExpertTrainer:

    def __init__(
            self,
            base_model_name, lora_rank, lora_alpha,
            lora_dropout, max_length
    ):
        self.model_name = base_model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_length = max_length

        # TODO: check if the tokenizer needs any special config or alternation
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, use_fast=True, padding_side='right', model_max_length=max_length
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # TODO: if we are going to use any other type of quantization, this should be parametrized
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=False,
        # )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            # quantization_config=self.bnb_config
        )
        self.base_model.enable_input_require_grads()

        # self.loftq_config = LoftQConfig(loftq_bits=4)
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            # loftq_config=self.loftq_config,
            target_modules=LORA_TARGET_MODULES,
            # modules_to_save=OTHER_TRAINABLE_MODULES,
            lora_dropout=self.lora_dropout,
            bias='none',
            task_type=TASK_TYPE
        )
        self.model = get_peft_model(self.base_model, self.lora_config)

        self.model.print_trainable_parameters()
        self.model.config.use_cache = False

    def train(self, train_data, eval_data, training_args):
        train_data = apply_preprocessing(train_data, create_message_column, self.tokenizer)
        eval_data = apply_preprocessing(eval_data, create_message_column, self.tokenizer)

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            dataset_text_field='text',
            max_seq_length=self.max_length,
            tokenizer=self.tokenizer,
            args=training_args,
        )
        trainer.train()
