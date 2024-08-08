import os
import random

import wandb
import numpy as np
import torch
from transformers import (
    TrainingArguments
)

from utils.arg_parser import train_arg_parser
from models.module_trainer import LoraModuleTrainer
from data_handler.dataset import read_dataset, get_data_collator, formatting_prompts_func


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = train_arg_parser()
    set_seed(args.seed)

    run_name = 'cluster' + str(args.cluster_idx) + '_batch' + str(args.batch_size) + '_prop' + str(args.data_portion)
    wandb.init(project=args.project_name, name=run_name)
    wandb.config.update(dict(vars(args)), allow_val_change=True)

    training_arguments = TrainingArguments(
        output_dir=args.output_dir + '/' + run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=False,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optim=args.optimizer,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        group_by_length=False,
        save_steps=args.save_every,
        logging_steps=args.log_every,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        report_to="wandb",
        run_name=run_name,  # TODO: Might be changed.
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=args.load_best_model_at_end,
        save_total_limit=args.save_total_limit
    )

    module_trainer = LoraModuleTrainer(
        base_model_name=args.model_name,
        lora_rank=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        formatting_func=formatting_prompts_func
    )

    train_data, eval_data = read_dataset(args.dataset_name, args.cluster_idx, args.data_portion, return_test=False)
    
    data_collator = get_data_collator(module_trainer.tokenizer)
    
    module_trainer.train(
        train_data=train_data, 
        eval_data=eval_data,  
        training_args=training_arguments,
        collator=data_collator
    )
