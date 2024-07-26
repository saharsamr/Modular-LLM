import wandb
import os
import pyarrow.dataset as pds
import pyarrow.compute as pc

from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from utils.arg_parser import arg_parser
from models.module_trainer import LoraModuleTrainer
from datasets import load_dataset



def read_dataset(ds_name, cluster_idx):
    """
    Returns the samples in the dataset based on the value of cluster_idx.
    (it is done inplace to not filling up the ram)

    - param1: dataset name from hf hub --> str
    - param2: index of the cluster that we want from the dataset --> str

    - return: the dataset containing all the samples with that specific cluster_idx
    """

    def effective_filter(ds, col_name, col_val):
        """
        This function effectively filters the dataset w.r.t a specific value for a column

        - param1: dataset
        - param2: column_name that we want to filter on that --> str
        - param3: specific column value that we want our instances have --> str 
        """
        expr = pc.field(col_name) == col_val

        filtered = ds.with_format("arrow").filter(
            lambda t: pds.dataset(t).to_table(columns={"mask": expr})[0].to_numpy(),
            batched=True,
        ).with_format(None)
        return filtered
    
    # Loading the whole dataset
    ds = load_dataset(ds_name)

    # Filtering the dataset based on the value of cluster_idx
    ds_filt_cl = effective_filter(ds, col_name='template_idx', col_val=cluster_idx)

    # Selecting the training rows
    train_ds = effective_filter(ds_filt_cl, col_name='split', col_val='train')

    # Selecting the validation rows
    val_ds = effective_filter(ds_filt_cl, col_name='split', col_val='validation')    

    return train_ds['train'], val_ds['train']

    

if __name__ == "__main__":
    args = arg_parser()

    wandb.login()
    os.environ["WANDB_PROJECT"] = "Modular-LLM"
    wandb.config.update(dict(vars(args)), allow_val_change=True)
    
    # Creating train_data and eval_data
    ds_name = "zhan1993/flan-10k-flat-cluster-embedding"
    train_data, eval_data = read_dataset(ds_name, args.cluster_idx)

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
        run_name=args.model_name + '_' + args.cluster_idx,  # TODO: Might be changed.
    )

    module_trainer = LoraModuleTrainer(
        base_model_name=args.model_name,
        lora_rank=args.rank,
        lora_alpha=args.lora_alpha,
    )
    module_trainer.train(
        train_data=train_data, 
        eval_data=eval_data,  
        training_args=training_arguments,
    )
