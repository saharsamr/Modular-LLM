from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
import re

bbh_subsets = [
    'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
    'formal_fallacies', 'geometric_shapes', 'hyperbaton',
    'logical_deduction_five_objects', 'logical_deduction_seven_objects',
    'logical_deduction_three_objects', 'movie_recommendation',
    'navigate', 'penguins_in_a_table', 'reasoning_about_colored_objects',
    'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding',
    'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects', 'web_of_lies',
    ]

ignored_bbh_subsets = [
    'dyck_languages', 'multistep_arithmetic_two', 'object_counting', 'word_sorting'
]


def read_test_dataset_lang(ds_name, lang):
    # lang: de (germany)
    if ds_name == 'arc-challenge':
        ds = load_dataset("alexandrainst/m_arc", lang, cache_dir='../../data/', split='train', trust_remote_code=True)
    elif ds_name == 'hswag':
        ds = load_dataset("alexandrainst/m_hellaswag", lang, cache_dir='../../data/', split='val', trust_remote_code=True)
    # elif ds_name == 'mgsm':
    #     ds = load_dataset("juletxara/mgsm", lang, cache_dir='../../data/', split='train', trust_remote_code=True)
    elif ds_name == 'xnli':
        ds = load_dataset("facebook/xnli", lang, cache_dir='../../data/', split='train', trust_remote_code=True)
    elif ds_name == 'mmlu':
        ds = load_dataset("alexandrainst/m_mmlu", lang, cache_dir='../../data/', split='train', trust_remote_code=True)
    else:  # xcopa xnli xquad xlsum  adamergx paper prompts
        raise f"Dataset {ds_name} is not supported yet."

    return ds


def extract_input_content_multilingual(ds_name, row):
    if ds_name == 'arc-challenge':
        return row['instruction']
    if ds_name == 'hswag':
        return row['ctx']
    if ds_name == 'xnli':
        return f"[premise]{row['premise']}[hypothesis]{row['hypothesis']}[relationship]"
    if ds_name == 'mmlu':
        return row['instruction']


def load_bbh_dataset():
    bbh_datasets = []
    for task in bbh_subsets:
        ds = load_dataset('maveriq/bigbenchhard', task, cache_dir='../data/', split='train', trust_remote_code=True)
        task_name_col = [task]*len(ds)
        ds = ds.add_column('task_name', task_name_col)
        bbh_datasets.append(ds)
    bbh = concatenate_datasets(bbh_datasets)
    return bbh


def read_test_dataset(ds_name):
    # https://huggingface.co/datasets/ybisk/piqa
    if ds_name == 'piqa':
        ds = load_dataset('ybisk/piqa', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/google/boolq
    elif ds_name == 'boolq':
        ds = load_dataset('google/boolq', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/allenai/swag
    elif ds_name == 'swag':
        ds = load_dataset('allenai/swag', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/Rowan/hellaswag?row=0
    elif ds_name == 'hswag':
        ds = load_dataset('Rowan/hellaswag', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/allenai/ai2_arc?row=10 -> consider test-split as well. 
    elif ds_name == 'arc-challenge':
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/allenai/ai2_arc?row=10 -> consider test-split as well. 
    elif ds_name == 'arc-easy':
        ds = load_dataset('allenai/ai2_arc', 'ARC-Easy', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/allenai/openbookqa?row=0
    elif ds_name == 'oqa':
        ds = load_dataset('allenai/openbookqa', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/maveriq/bigbenchhard
    elif ds_name == 'bbh':
        ds = load_bbh_dataset()
    # https://huggingface.co/datasets/allenai/winogrande -> consider chaning the input format
    elif ds_name == 'wg':
        ds = load_dataset('allenai/winogrande', 'winogrande_xl', cache_dir='../data/', split='validation', trust_remote_code=True)
    # https://huggingface.co/datasets/openai/openai_humaneval
    elif ds_name == 'he':
        ds = load_dataset('openai/openai_humaneval', cache_dir='../data/', split='test', trust_remote_code=True)
    # https://huggingface.co/datasets/google-research-datasets/mbpp
    elif ds_name == 'mbpp':
        ds = load_dataset('google-research-datasets/mbpp', cache_dir='../data/', split='full', trust_remote_code=True)
    elif ds_name == 'flan':
        ds = load_dataset("TahaBa/flan-routing-MoE-dataset", cache_dir="../data/", trust_remote_code=True)['test']
    else:
        raise f"Dataset {ds_name} is not supported yet."

    return ds


def extract_input_content(ds_name, rows):
    if ds_name == 'piqa':
        return rows['goal']
    if ds_name == 'boolq':
        return [f'[passage]{passage}[question]{question}' for passage, question in zip(rows['passage'], rows['question'])]
    if ds_name == 'swag':
        return rows['startphrase']
    if ds_name == 'hswag':
        return [f'{activity}: {context}' for activity, context in zip(rows['activity_label'], rows['ctx'])]
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        return rows['question']
    if ds_name == 'oqa':
        return rows['question_stem']
    if ds_name == 'bbh':
        return rows['input']
    if ds_name == 'wg':
        return rows['sentence']
    if ds_name == 'flan':
        return f"{rows['source']}"


def create_multi_choice_options_multilingual(row, ds_name):
    options_texts = []
    content = extract_input_content_multilingual(ds_name, row)
    if ds_name == 'hswag':
        choices = row['endings']
    if ds_name == 'arc-challenge':
        choices = str([row['option_a'], row['option_b'], row['option_c'], row['option_d']])
    if ds_name == 'xnli':
        choices = str(['entailment', 'neutral', 'contradiction'])
    if ds_name == 'mmlu':
        choices = str([row['option_a'], row['option_b'], row['option_c'], row['option_d']])


    for choice in choices:
        options_texts.append(f'<|user|>\n{content}<|end|>\n<|assistant|>{choice}<|end|>\n')

    return options_texts


def get_bbh_options(rows):
    batch_choices = []
    for row_input, row_task in zip(rows['input'], rows['task_name']):
        if row_task == 'boolean_expressions':
            choices = ['True', 'False']
        elif (row_task == 'causal_judgement') or (row_task == 'navigate') or (row_task == 'web_of_lies'):
            choices = ['Yes', 'No']
        elif row_task == 'formal_fallacies':
            choices = ['valid', 'invalid']
        elif row_task == 'sports_understanding':
            choices = ['yes', 'no']
        elif row_task in bbh_subsets:
            choices = re.findall(r'\([A-Z]\)', row_input)
        else:
            raise 'This subset is not supported'

        batch_choices.append(choices)

    return batch_choices


def create_multi_choice_options(rows, ds_name, tokenizer):
    batch_options = []
    contents = extract_input_content(ds_name, rows)
    if ds_name == 'piqa':
        choices = [[sol1, sol2] for sol1, sol2 in zip(rows['sol1'], rows['sol2'])]
    if ds_name == 'boolq':
        choices = [['true', 'false'] for _ in rows['passage']]
    if ds_name == 'swag':
        choices = [[e1, e2, e3, e4] for e1, e2, e3, e4 in zip(rows['ending0'], rows['ending1'], rows['ending2'], rows['ending3'])]
    if ds_name == 'hswag':
        choices = [[e[i] for e in rows['endings']] for i in range(len(rows['endings'][0]))]
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        choices = [[opts[i] for opts in rows['choices']['text']] for i in range(len(rows['choices']['text'][0]))]
    if ds_name == 'wg':
        choices = [[option1, option2] for option1, option2 in zip(rows['option1'], rows['option2'])]
    if ds_name == 'oqa':
        choices = [[opts[i] for opts in rows['choices']['text']] for i in range(len(rows['choices']['text'][0]))]
    if ds_name == 'bbh':
        choices = get_bbh_options(rows)

    batch_options = []
    for sample_choices, content in zip(choices, contents):
        sample_options = []
        for choice in sample_choices:
            chat = [
                {'role': 'user', 'content': content},
                {'role': 'assistant', 'content': choice}
            ]
            sample_options.append(tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False))
        batch_options.append(sample_options)

    return batch_options


def extract_multi_choice_target_index_multilingual(row, ds_name):
    if ds_name == 'hswag':
        return int(row['label'])
    if ds_name == 'arc-challenge':
        return ['A', 'B', 'C', 'D'].index(row['answer'])
    if ds_name == 'xnli':
        return int(row['label'])
    if ds_name == 'mmlu':
        return ['A', 'B', 'C', 'D'].index(row['answer'])


def extract_multi_choice_target_index(rows, ds_name):
    if ds_name == 'piqa':
        return [int(target) for target in rows['label']]
    if ds_name == 'boolq':
        return [0 if ans == True else 1 for ans in rows['answer']]
    if ds_name == 'swag':
        return [int(lbl) for lbl in rows['label']]
    if ds_name == 'hswag':
        return [int(lbl) for lbl in rows['label']]
    if (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        return [[ch[i] for ch in rows['choices']['label']].index(ans) for i, ans in enumerate(rows['answerKey'])]
    if ds_name == 'wg':
        return [int(ans) - 1 for ans in rows['answer']]
    if ds_name == 'oqa':
        return [[ch[i] for ch in rows['choices']['label']].index(ans) for i, ans in enumerate(rows['answerKey'])]
    if ds_name == 'bbh':
        choices = get_bbh_options(rows)
        return [choice.index(target) for choice, target in zip(choices, rows['target'])]


def split_dataset_by_option_count(ds, ds_name):
    if ds_name in ['piqa', 'boolq', 'wg', 'swag']:
        return [ds]

    if ds_name == 'hswag':
        option_count = [len(endings) for endings in ds['endings']]
    elif (ds_name == 'arc-challenge') or (ds_name == 'arc-easy'):
        option_count = [len(choices['label']) for choices in ds['choices']]
    elif ds_name == 'oqa':
        option_count = [len(choices['label']) for choices in ds['choices']]
    elif ds_name == 'bbh':
        options = get_bbh_options(ds)
        option_count = [len(sample_options) for sample_options in options]
    else:
        raise "Pass a supported dataset"

    ds = ds.add_column('option_count', option_count)
    option_count_values = list(set(ds['option_count']))

    ds_list = []
    for option_count in option_count_values:
        ds_list.append(ds.filter(lambda sample: sample['option_count'] == option_count))
    return ds_list
