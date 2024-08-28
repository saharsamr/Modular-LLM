from merging_lora_modules.base_merging_module import (
    BaseMergingModule,
    cluster_checkpoint_names,
)
from transformers import AutoConfig, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

import xlora
import torch
from data_handler.dataset import (
    create_message_column,
    apply_preprocessing,
)


class XLoraAveraging(BaseMergingModule):
    def __init__(self, base_model, tokenizer, model_name):
        super().__init__(base_model, tokenizer, model_name)

    def merge(self, load_path=None):
        if load_path:
            self.base_model = xlora.from_pretrained(
                load_path, self.base_model, cluster_checkpoint_names
            )
        else:
            self.load_lora_modules()
            self.train()

    def load_lora_modules(self):
        self.base_model.config.use_cache = False
        self.base_model = xlora.add_xlora_to_model(
            model=self.base_model,
            xlora_config=xlora.xLoRAConfig(
                self.base_model_config.hidden_size,
                base_model_id=self.model_name,
                xlora_depth=4,
                device=torch.device("cuda"),
                adapters=cluster_checkpoint_names,
            ),
            verbose=True,
        )

    def train(self):
        routing_dataset = load_dataset("TahaBa/flan-routing-MoE-dataset", cache_dir="../data/")

        routing_train_dataset = apply_preprocessing(routing_dataset['train'], create_message_column, self.tokenizer)
        routing_validation_dataset = apply_preprocessing(
            routing_dataset['validation'], create_message_column, self.tokenizer)

        training_arguments = TrainingArguments(
            output_dir='./results/MoE-XLoRA',
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            max_grad_norm=2.0,
            gradient_checkpointing=True,
            learning_rate=1e-4,
            optim='paged_adamw_8bit',
            lr_scheduler_type='linear',
            group_by_length=False,
            save_steps=200,
            logging_steps=100,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=200,
            report_to="wandb",
            run_name='MoE-XLoRA',  # TODO: Might be changed.
            load_best_model_at_end=True,
            save_total_limit=2
        )

        trainer = SFTTrainer(
            model=self.base_model,
            train_dataset=routing_train_dataset,
            eval_dataset=routing_validation_dataset,
            dataset_text_field='text',
            max_seq_length=self.max_length,
            tokenizer=self.tokenizer,
            args=training_arguments,
        )
        trainer.train()
