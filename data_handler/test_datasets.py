from datasets import load_dataset, concatenate_datasets, DatasetDict


def read_test_dataset(ds_name):
    # https://huggingface.co/datasets/ybisk/piqa
    if ds_name == 'piqa':
        ds = load_dataset('ybisk/piqa', cache_dir='../data/', split='train', trust_remote_code=True)
    # https://huggingface.co/datasets/google/boolq
    elif ds_name == 'boolq':
        ds = load_dataset('google/boolq', cache_dir='../data/', split='train', trust_remote_code=True)
    # https://huggingface.co/datasets/allenai/swag
    elif ds_name == 'swag':
        ds = load_dataset('allenai/swag', cache_dir='../data/', split='train', trust_remote_code=True)
    # https://huggingface.co/datasets/Rowan/hellaswag?row=0
    elif ds_name == 'hswag':
        ds = load_dataset('Rowan/hellaswag', cache_dir='../data/', split='train', trust_remote_code=True)
    # https://huggingface.co/datasets/allenai/ai2_arc?row=10
    elif ds_name == 'arc-challenge':
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', cache_dir='../data/', split='train', trust_remote_code=True)
    # https://huggingface.co/datasets/allenai/ai2_arc?row=10
    elif ds_name == 'arc-easy':
        ds = load_dataset('allenai/ai2_arc', 'ARC-Easy', cache_dir='../data/', split='train', trust_remote_code=True)
    # https://huggingface.co/datasets/m-rousseau/oqa-v1?row=68
    elif ds_name == 'oqa':
        ds = load_dataset('m-rousseau/oqa-v1', cache_dir='../data/', split='train', trust_remote_code=True)
    # https://huggingface.co/datasets/SaylorTwift/bbh
    elif ds_name == 'bbh':
        ds = load_dataset('SaylorTwift/bbh', cache_dir='../data/', split='train', trust_remote_code=True)
    elif ds_name == 'flan':
        ds = load_dataset("TahaBa/flan-routing-MoE-dataset", cache_dir="../data/", trust_remote_code=True)['test']
    else:
        raise f"Dataset {ds_name} is not supported yet."

    return ds


def extract_input_content(ds_name, row):
    if ds_name == 'piqa':
        return row['goal']
    if ds_name == 'boolq':
        return f"[passage]{row['passage']}[question]{row['question']}"
    if ds_name == 'swag':
        return row['startphrase']
    if ds_name == 'hswag':
        return row['ctx']
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        return row['question']
    # if ds_name == 'oqa':
    #     return f"[context]{row['context']}\n[question]{row['question']}\n"
    # if ds_name == 'bbh':
    #     return f"[question]{row['input']}\n"
    if ds_name == 'flan':
        return f"{row['source']}"


def create_multi_choice_options(row, ds_name):
    options_texts = []
    content = extract_input_content(ds_name, row)
    if ds_name == 'piqa':
        choices = [row['sol1'], row['sol2']]
    if ds_name == 'boolq':
        choices = ['true', 'false']
    if ds_name == 'swag':
        choices = [row['ending0'], row['ending1'], row['ending2'], row['ending3']]
    if ds_name == 'hswag':
        choices = row['endings']
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        choices = row['choices']['text']

    for choice in choices:
        options_texts.append(f'<|user|>\n{content}<|end|>\n<|assistant|>{choice}<|end|>\n')

    return options_texts


def extract_multi_choice_target_index(row, ds_name):
    if ds_name == 'piqa':
        return int(row['label'])
    if ds_name == 'boolq':
        return 0 if row['answer'] is True else 1
    if ds_name == 'swag':
        return int(row['label'])
    if ds_name == 'hswag':
        return int(row['label'])
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        if row['answerKey'] == 'A':
            return 0
        elif row['answerKey'] == 'B':
            return 1
        elif row['answerKey'] == 'C':
            return 2
        elif row['answerKey'] == 'D':
            return 3
        else:
            raise 'More than 4 options in ARC dataset.'

