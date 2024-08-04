from datasets import load_dataset
import pyarrow.dataset as pds
import pyarrow.compute as pc
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils.config import *


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
    ds = load_dataset(ds_name, cache_dir="../data/")

    # Filtering the dataset based on the value of cluster_idx
    ds_filt_cl = effective_filter(ds, col_name='template_idx', col_val=cluster_idx)

    # Selecting the training rows
    train_ds = effective_filter(ds_filt_cl, col_name='split', col_val='train')['train']

    # Selecting the validation rows
    val_ds = effective_filter(ds_filt_cl, col_name='split', col_val='validation')['train']

    return train_ds, val_ds


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['source'])):
        max_source_length = (MAX_LENGTH - len(example['target'][i].split())) / AVG_WORD_TOKEN
        text = f"### Instruction: {example['source'][i][:max_source_length]}\n ### Response: {example['target'][i]}"
        output_texts.append(text)
    return output_texts


def get_data_collator(tokenizer):
    response_template = " ### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return collator
