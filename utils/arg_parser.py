import argparse
from utils.config import *


def experts_training_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--lang_independent", action="store_true", help="Whether or not to train experts language independent.")
    parser.add_argument("--lang_expert_path", type=str, default='../data/hub/models--AliEdalat--le_en_1.8k_train_5token_pred/snapshots/402576cae80a80040dcaeb3fc7406e9f6c0b0371/')
    parser.add_argument("--dataset_name", type=str, default="TahaBa/flan-10K-cluster-splitted")
    parser.add_argument("--project_name", type=str, default='TrainExperts')
    parser.add_argument("--cluster_idx", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default='./results')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--optimizer", default="paged_adamw_8bit")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=int, default=0.03)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--metric_for_best_model", type=str, default='eval_loss')
    parser.add_argument("--load_best_model_at_end", type=bool, default='True')
    parser.add_argument("--save_total_limit", type=int, default=2)

    return parser.parse_args()


def experts_testing_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset_name", type=str, default="TahaBa/flan-10K-cluster-splitted")
    parser.add_argument("--project_name", type=str, default='Modular-LLM')
    parser.add_argument("--cluster_idx", type=int, required=True)  # The index of cluster that we want to train
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--no_lora", action="store_true")

    parser.add_argument("--posthoc_cross_lingual", action="store_true")
    parser.add_argument("--source_formal_expert_path", type=str,
                        default="results/cluster0_batch16_prop1.0_langen_5000/checkpoint-17/")
    parser.add_argument("--target_formal_expert_path", type=str,
                        default="results/cluster0_batch16_prop1.0_langger_5000/checkpoint-17/")
    parser.add_argument("--disentanglement_method", type=str, choices=['subtract', 'orthogonal_projection'])
    parser.add_argument("--add_functional_only", type=bool, default=False)

    return parser.parse_args()


def experts_merging_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--project_name", type=str, default='Modular-LLM')
    parser.add_argument(
        "--merging_strategy", type=str, required=True,
        choices=['simple_average', 'xlora_average', 'arrow_routing', 'phi3']
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        choices=[
            'piqa', 'boolq', 'swag', 'hswag', 'arc-challenge', 'arc-easy', 'oqa', 'bbh', 'flan', 'wg'
        ]
    )
    parser.add_argument("--test_type", type=str, default='zero_shot', choices=['zero_shot', 'few_shot'])

    # cross_lingual parser arguments
    parser.add_argument("--posthoc_cross_lingual", action="store_true")
    parser.add_argument("--target_lang", type=str, default='de')
    parser.add_argument("--factorize_average_lora", action="store_true")
    parser.add_argument("--source_formal_expert_path", type=str, default="results/cluster0_batch16_prop1.0_langen_5000/checkpoint-17/")
    parser.add_argument("--target_formal_expert_path", type=str, default="results/cluster0_batch16_prop1.0_langger_5000/checkpoint-17/")
    parser.add_argument("--disentanglement_method", type=str, choices=['subtract', 'orthogonal_projection'])
    parser.add_argument("--add_functional_only", type=bool, default=False)

    return parser.parse_args()
