from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from utils.arg_parser import arg_parser
from models.module_trainer import LoraModuleTrainer

import wandb
import os


if __name__ == "__main__":
    args = arg_parser()

    wandb.login()
    os.environ["WANDB_PROJECT"] = "Modular-LLM"


    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optim=args.optimizer,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ration,
        group_by_length=True,
        save_steps=args.save_every,
        logging_steps=args.log_every,
        report_to="wandb",
        run_name="",  # TODO: define run-name based on datasets, expert we are going to train, etc.
    )

    module_trainer = LoraModuleTrainer(
        base_model_name=args.model_name,
        lora_rank=args.rank,
        lora_alpha=args.lora_alpha,
    )
    module_trainer.train(
        train_data=None,  # TODO: define train data
        eval_data=None,  # TODO: define eval data
        training_args=training_arguments,
    )