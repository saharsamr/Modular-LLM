from datasets import load_dataset, concatenate_datasets, DatasetDict
import pyarrow.dataset as pds
import pyarrow.compute as pc
import random


def effective_filter(ds, expr):
    """
    This function effectively filters the dataset w.r.t an expression

    - param1: dataset
    - param2: expression
    
    - return: filtered dataset based on expression
    """

    filtered = ds.with_format("arrow").filter(
        lambda t: pds.dataset(t).to_table(columns={"mask": expr})[0].to_numpy(),
        batched=True,
    ).with_format(None)
    return filtered


def sample_from_each_cluster(ds, sample_num):
    """
    sample randomly from each cluster

    - param1: dataset
    - param2: number of sample per batch

    - return: final concatenated dataset
    """
    ds_list = [] 
    for i in range(10):
        ds_filt_cl = effective_filter(ds, expr = pc.field('template_idx') == i)
        random_indices = random.sample(range(ds_filt_cl.num_rows), k=sample_num)
        ds_filt_cl_sample = ds_filt_cl.select(random_indices)
        ds_list.append(ds_filt_cl_sample)
    
    sampled_concat_ds = concatenate_datasets(ds_list)

    return sampled_concat_ds


def create_and_push_flan_dataset(ds_name):

    """
    Returns the samples in the dataset based on the value of cluster_idx.
    (it is done inplace to not filling up the ram)

    - param1: dataset name from hf hub --> str
    - param2: index of the cluster that we want from the dataset --> str

    - return: the dataset containing all the samples with that specific cluster_idx
    """

    ds = load_dataset(ds_name, split="train")

    train_ds = effective_filter(ds, expr = pc.field('split') == 'train')
    val_ds = effective_filter(ds, expr = pc.field('split') == 'validation')
    test_ds = effective_filter(ds, expr = pc.field('split') == 'test')

    train_ds = train_ds.remove_columns("split")
    val_ds = val_ds.remove_columns("split")
    test_ds = test_ds.remove_columns("split")

    final_dd = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

    final_dd.push_to_hub("TahaBa/flan-10K-cluster-splitted", private=True)

    return final_dd


def create_and_push_routing_flan_dataset(ds_name):
    """
    Creating a dataset from flan, suitable for routing

    - param1: dataset name from hf hub --> str
    - param2: number of samples from each first 10 clusters

    - return: the routing dataset
    """

    ds = load_dataset(ds_name)
    train_ds = ds['train']
    val_ds = ds['validation']
    test_ds = ds['test']
    
    # 8, 1, 1 ratio
    sampled_train_ds = sample_from_each_cluster(train_ds, 2000)
    sampled_val_ds = sample_from_each_cluster(val_ds, 250)
    sampled_test_ds = sample_from_each_cluster(test_ds, 250)

    # Adding the samples which are not within 10 clusters
    out_cluster_train_ds = effective_filter(train_ds, expr = pc.field('template_idx') >= 10)
    final_train_ds = concatenate_datasets([sampled_train_ds, out_cluster_train_ds])
    final_train_ds = final_train_ds.shuffle(seed=42)

    out_cluster_val_ds = effective_filter(val_ds, expr = pc.field('template_idx') >= 10)
    final_val_ds = concatenate_datasets([sampled_val_ds, out_cluster_val_ds])
    final_val_ds = final_val_ds.shuffle(seed=42)

    out_cluster_test_ds = effective_filter(test_ds, expr = pc.field('template_idx') >= 10)
    final_test_ds = concatenate_datasets([sampled_test_ds, out_cluster_test_ds])
    final_test_ds = final_test_ds.shuffle(seed=42)

    final_dd = DatasetDict({'train': final_train_ds, 'validation': final_val_ds, 'test': final_test_ds})

    final_dd.push_to_hub("TahaBa/flan-routing-MoE-dataset", private=True)

    return final_dd


def read_dataset(ds_name, cluster_idx, data_portion, return_test):

    """
    Returns the samples in the dataset based on the value of cluster_idx.
    (it is done inplace to not filling up the ram)

    - param1: dataset name from hf hub --> str
    - param2: index of the cluster that we want from the dataset --> str
    - param3: data proportion
    - param4: flag for returning test dataset

    - return: the dataset containing all the samples with that specific cluster_idx
    """

    ds = load_dataset(ds_name, cache_dir="../data/")

    if return_test:
        test_ds = effective_filter(ds['test'], pc.field('template_idx') == cluster_idx)
        test_ds = test_ds if data_portion == 1.0 else test_ds.train_test_split(test_size=1-data_portion)['train']

        return test_ds

    train_ds = effective_filter(ds['train'], pc.field('template_idx') == cluster_idx)
    train_ds = train_ds if data_portion == 1.0 else train_ds.train_test_split(test_size=1-data_portion)['train']

    val_ds = effective_filter(ds['validation'], pc.field('template_idx') == cluster_idx)
    val_ds = val_ds if data_portion == 1.0 else val_ds.train_test_split(test_size=1-data_portion)['train']

    return train_ds, val_ds


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


def apply_preprocessing(data, prompt_func, tokenizer):
    data = data.map(prompt_func)
    # tokenizer.chat_template="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    data = data.map(
        lambda sample:
        {"text": tokenizer.apply_chat_template(
            sample["messages"],
            # chat_template="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            add_generation_prompt=False, tokenize=False)}
    )
    print(data)
    print(data['text'][0])
    return data
