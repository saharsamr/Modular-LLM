from datasets import load_dataset
import pyarrow.dataset as pds
import pyarrow.compute as pc
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils.config import *


def read_dataset(ds_name, cluster_idx, data_portion, return_test):

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

    ds = load_dataset(ds_name, cache_dir="../data/", split="train")
    ds_filt_cl = effective_filter(ds, col_name='template_idx', col_val=cluster_idx)

    if return_test:
        test_ds = effective_filter(ds_filt_cl, col_name='split', col_val='test')
        test_ds = test_ds.train_test_split(test_size=1-data_portion)['train']
        test_ds = test_ds.map(prompt_func_test, batched=True)

        return test_ds

    train_ds = effective_filter(ds_filt_cl, col_name='split', col_val='train')
    train_ds = train_ds.train_test_split(test_size=1-data_portion)['train']

    val_ds = effective_filter(ds_filt_cl, col_name='split', col_val='validation')
    val_ds = val_ds.train_test_split(test_size=1-data_portion)['train']

    return train_ds, val_ds


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['source'])):
        max_source_length = int((MAX_LENGTH - len(example['target'][i].split())) / AVG_WORD_TOKEN)
        source = ' '.join(example['source'][i].split()[:max_source_length])
        source = source if source[-1] == '.' else source + '.'
        text = f"Instruct: {source}\nOutput: {example['target'][i]}"
        output_texts.append(text)
    return output_texts


def get_data_collator(tokenizer):
    response_template = "Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return collator


def prompt_func_test(example):
    output_texts = []
    for i in range(len(example['source'])):
        max_source_length = int(MAX_SOURCE_TOKENS / AVG_WORD_TOKEN)
        source = ' '.join(example['source'][i].split()[:max_source_length])
        source = source if source[-1] == '.' else source + '.'
        text = f"Instruct: {source}\nOutput: "
        output_texts.append(text)
    return {'source': output_texts}
