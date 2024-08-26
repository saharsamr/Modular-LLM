from datasets import load_dataset, concatenate_datasets
import pyarrow.dataset as pds
import pyarrow.compute as pc
import random


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


def read_dataset(ds_name, cluster_idx, data_portion, return_test):

    """
    Returns the samples in the dataset based on the value of cluster_idx.
    (it is done inplace to not filling up the ram)

    - param1: dataset name from hf hub --> str
    - param2: index of the cluster that we want from the dataset --> str

    - return: the dataset containing all the samples with that specific cluster_idx
    """

    ds = load_dataset(ds_name, cache_dir="../data/", split="train")
    ds_filt_cl = effective_filter(ds, col_name='template_idx', col_val=cluster_idx)

    if return_test:
        test_ds = effective_filter(ds_filt_cl, col_name='split', col_val='test')
        test_ds = test_ds if data_portion == 1.0 else test_ds.train_test_split(test_size=1-data_portion)['train']

        return test_ds

    train_ds = effective_filter(ds_filt_cl, col_name='split', col_val='train')
    train_ds = train_ds if data_portion == 1.0 else train_ds.train_test_split(test_size=1-data_portion)['train']

    val_ds = effective_filter(ds_filt_cl, col_name='split', col_val='validation')
    val_ds = val_ds if data_portion == 1.0 else val_ds.train_test_split(test_size=1-data_portion)['train']

    return train_ds, val_ds


def sample_from_each_cluster(ds, sample_num):
    ds_list = [] 
    for i in range(10):
        ds_filt_cl = effective_filter(ds, col_name='template_idx', col_val=i)
        random_indices = random.sample(range(ds_filt_cl.num_rows), k=sample_num)
        ds_filt_cl_sample = ds_filt_cl.select(random_indices)
        ds_list.append(ds_filt_cl_sample)
    
    sampled_concat_ds = concatenate_datasets(ds_list)

    return sampled_concat_ds


def read_routing_ds_flan(ds_name):
    """
    Creating a dataset from flan, suitable for routing

    - param1: dataset name from hf hub --> str

    - return: the routing dataset
    """

    ds = load_dataset(ds_name, cache_dir="../data/", split="train")

    train_ds = effective_filter(ds, col_name='split', col_val='train')
    val_ds = effective_filter(ds, col_name='split', col_val='validation')

    sampled_train_ds = sample_from_each_cluster(train_ds, 500)
    sampled_val_ds = sample_from_each_cluster(val_ds, 75)

    # Now we go for other 10 clusters
    # We shouldn't sample cluster by cluster this time, we sample from all of them



def create_message_column(row):
    messages = []
    user = {
        "content": f"{row['source']}",
        "role": "user"
    }
    messages.append(user)
    assistant = {
        "content": f"{row['target']}",
        "role": "assistant"
    }
    messages.append(assistant)
    return {"messages": messages}


def create_message_column_for_test(row):
    messages = []
    user = {
        "content": f"{row['source']}",
        "role": "user"
    }
    messages.append(user)
    return {"messages": messages}
