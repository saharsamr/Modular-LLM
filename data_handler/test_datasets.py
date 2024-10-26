from datasets import load_dataset, concatenate_datasets, DatasetDict


def read_test_dataset(ds_name):
    # https://huggingface.co/datasets/ybisk/piqa
    if ds_name == 'piqa':
        ds = load_dataset('ybisk/piqa', cache_dir='../data/', split='train')
    # https://huggingface.co/datasets/google/boolq
    elif ds_name == 'boolq':
        ds = load_dataset('google/boolq', cache_dir='../data/', split='train')
    # https://huggingface.co/datasets/allenai/swag
    elif ds_name == 'swag':
        ds = load_dataset('allenai/swag', cache_dir='../data/', split='train')
    # https://huggingface.co/datasets/allenai/ai2_arc?row=10
    elif ds_name == 'arc-challenge':
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', cache_dir='../data/', split='train')
    # https://huggingface.co/datasets/allenai/ai2_arc?row=10
    elif ds_name == 'arc-easy':
        ds = load_dataset('allenai/ai2_arc', 'ARC-Easy', cache_dir='../data/', split='train')
    # https://huggingface.co/datasets/m-rousseau/oqa-v1?row=68
    elif ds_name == 'oqa':
        ds = load_dataset('m-rousseau/oqa-v1', cache_dir='../data/', split='train')
    # https://huggingface.co/datasets/SaylorTwift/bbh
    elif ds_name == 'bbh':
        ds = load_dataset('SaylorTwift/bbh', cache_dir='../data/', split='train')
    elif ds_name == 'flan':
        ds = load_dataset("TahaBa/flan-routing-MoE-dataset", cache_dir="../data/")['test']
    else:
        raise f"Dataset {ds_name} is not supported yet."

    return ds


def create_user_content(ds_name, row):
    if ds_name == 'piqa':
        return f"[question]{row['goal']}\n'sol1'{row['sol1']}\n'sol2'{row['sol2']}\n"
    if ds_name == 'boolq':
        return f"[passage]{row['passage']}\n[question]{row['question']}\n"
    if ds_name == 'swag':
        return (f"[start phrase]{row['startphrase']}\n[ending0]{row['ending0']}\n[ending1]{row['ending1']}\n"
                f"[ending2]{row['ending2']}\n[ending3]{row['ending3']}\n")
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        choices = '\n'.join(
            [f'{label}) {text}' for label, text in zip(row['choices']['label'], row['choices']['text'])]
        )
        return f"[question]{row['question']}\n{choices}\n"
    if ds_name == 'oqa':
        return f"[context]{row['context']}\n[question]{row['question']}\n"
    if ds_name == 'bbh':
        return f"[question]{row['input']}\n"
    if ds_name == 'flan':
        return f"{row['source']}"


def create_assistant_target(ds_name, row):
    if ds_name == 'piqa':
        return f"{row['label']}"
    if ds_name == 'boolq':
        return f"{row['answer']}"
    if ds_name == 'swag':
        return f"{row['label']}"
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        return f"{row['answerKey']}"
    if ds_name == 'oqa':
        return f"{row['answers']['text'][0]}"
    if ds_name == 'bbh':
        return f"{row['target']}\n"
    if ds_name == 'flan':
        return f"{row['target']}\n"


def create_few_shot_message(sample_rows, ds_name):
    messages = []
    for row in sample_rows[:-1]:
        user = {
            "content": create_user_content(ds_name, row),
            "role": "user"
        }
        messages.append(user)
        assistant = {
            "content": create_assistant_target(ds_name, row),
            "role": "assistant"
        }
        messages.append(assistant)
    messages.append({
        "content": create_user_content(ds_name, sample_rows[-1]),
        "role": "user"
    })
    return {"messages": messages}


def create_zero_shot_message(row, ds_name):
    messages = []
    user = {
        "content": create_user_content(ds_name, row),
        "role": "user"
    }
    messages.append(user)
    return {"messages": messages}
