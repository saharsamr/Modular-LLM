import random

import wandb
import numpy as np
import torch
from transformers import TrainingArguments

from utils.arg_parser import experts_training_arg_parser
from models.expert_trainer import ExpertTrainer
from models.lang_independent_expert_trainer import LangIndependentExpertTrainer
from data_handler.dataset import read_dataset
from utils.config import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = experts_training_arg_parser()
    set_seed(args.seed)

    if args.lang_independent:
        run_name = 'llama/lang_ind_cluster' + str(args.cluster_idx) + '_batch' + str(args.batch_size) + '_prop' + str(args.data_portion)
    else:
        run_name = 'llama/cluster' + str(args.cluster_idx) + '_batch' + str(args.batch_size) + '_prop' + str(args.data_portion)
    wandb.init(project=args.project_name, name=run_name)
    wandb.config.update(dict(vars(args)), allow_val_change=True)

    training_arguments = TrainingArguments(
        output_dir=args.output_dir + '/' + run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=2.0,
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
        load_best_model_at_end=args.load_best_model_at_end,
        save_total_limit=args.save_total_limit
    )

    if args.lang_independent:
        module_trainer = LangIndependentExpertTrainer(
            base_model_name=args.model_name,
            lang_expert_path=args.lang_expert_path,
            lora_rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_length=MAX_LENGTH
        )
    else:
        module_trainer = ExpertTrainer(
            base_model_name=args.model_name,
            lora_rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_length=MAX_LENGTH
        )

    train_data, eval_data = read_dataset(args.dataset_name, args.cluster_idx, args.data_portion, return_test=False)

    module_trainer.train(
        train_data=train_data,
        eval_data=eval_data,
        training_args=training_arguments
    )
