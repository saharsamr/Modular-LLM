import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    LoftQConfig,
    PeftModel
)


class LoraModuleTrainer:

    def __init__(
            self,
            base_model_name,
            lora_rank, lora_alpha, lora_dropout,
    ):
        self.model_name = base_model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # TODO: check if the tokenizer needs any special config or alternation
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # TODO: if we are going to use any other type of quantization, this should be parametrized
        """
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,  # TODO: what dtype?
            bnb_4bit_use_double_quant=False,
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,  # TODO: what dtype?
            quantization_config=self.bnb_config
        )
        
        self.loftq_config = LoftQConfig(loftq_bits=4)
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            loftq_config=self.loftq_config,
            target_modules='all-linear',  # TODO: what are the target modules exactly?
            modules_to_save=['lm_head', 'embed_tokens'],  # TODO: what other modules should be trainable?
            lora_dropout=self.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        self.model = get_peft_model(self.base_model, self.lora_config)
        """

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float32,  # you may change it with different models
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,  # float32 is tested and veryfied
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
        )

        # In case of using gradient_checkpoint = True, the below line should be used:
        base_model.enable_input_require_grads()

        self.model = PeftModel.from_pretrained(
                            base_model,
                            base_model_name,
                            subfolder="loftq_init",
                            is_trainable=True,
                            )

        self.model.print_trainable_parameters()
        self.model.config.use_cache = False



    def train(self, train_data, eval_data, training_args):

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            # data_collator=data_collator,  # TODO: define data-collator
            train_dataset=train_data,
            eval_dataset=eval_data
        )

        trainer.train()
