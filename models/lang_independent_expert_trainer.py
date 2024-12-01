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

from models.expert_trainer import ExpertTrainer
from data_handler.dataset import (
    create_message_column,
    apply_preprocessing
)
from utils.config import *


class LangIndependentSFTTrainer(SFTTrainer):
    def __init__(self, lang_expert_model, dot_product_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lang_expert_model = lang_expert_model
        self.dot_product_weight = dot_product_weight
        
        self.lang_expert_lora_params = {
            name: param.detach()
            for name, param in lang_expert_model.named_parameters()
            if 'lora' in name
        }

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        current_lora_params = {
            name: param
            for name, param in model.named_parameters()
            if 'lora' in name
        }
        
        inner_product = 0.0
        for name, param in current_lora_params.items():
            if name.replace('default', 'lang_expert') in self.lang_expert_lora_params.keys():
                lang_param = self.lang_expert_lora_params[name.replace('default', 'lang_expert')]
                inner_product += (param * lang_param).sum()
                
        loss += self.dot_product_weight * inner_product
        if return_outputs:
            return (loss, outputs)
        else:
            return loss


class LangIndependentExpertTrainer(ExpertTrainer):

    def __init__(
            self,
            base_model_name, lang_expert_path, lora_rank, lora_alpha,
            lora_dropout, max_length
    ):
        super().__init__(
            base_model_name, lora_rank, lora_alpha,
            lora_dropout, max_length
        )
        self.lang_expert_path = lang_expert_path


    def get_lang_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, padding_side='right', model_max_length=MAX_LENGTH
        )
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print('Loading Model ...')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        lang_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, quantization_config=bnb_config)
        
        lang_model = PeftModel.from_pretrained(lang_model, self.lang_expert_path, adapter_name='lang_expert').to('cuda')
        return lang_model

    def train(self, train_data, eval_data, training_args):
        train_data = apply_preprocessing(train_data, create_message_column, self.tokenizer)
        eval_data = apply_preprocessing(eval_data, create_message_column, self.tokenizer)

        trainer = LangIndependentSFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            dataset_text_field='text',
            max_seq_length=self.max_length,
            tokenizer=self.tokenizer,
            args=training_args,
            lang_expert_model=self.get_lang_model(),
            dot_product_weight=0.01
        )
        trainer.train()
