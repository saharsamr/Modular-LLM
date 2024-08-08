import argparse


def train_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default='zhan1993/flan-10k-flat-cluster-embedding')
    parser.add_argument("--project_name", type=str, default='Modular-LLM')
    parser.add_argument("--cluster_idx", type=int, required=True)  # The index of cluster that we want to train
    parser.add_argument("--output_dir", type=str, default='./results')
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--optimizer", default="paged_adamw_8bit")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=int, default=0.03)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    # parser.add_argument("--save_dir", type=str, default='./checkpoints')
    # parser.add_argument("--max_checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--metric_for_best_model", type=str, default='eval_loss')
    parser.add_argument("--load_best_model_at_end", type=bool, default='True')
    parser.add_argument("--save_total_limit", type=int, default=1)

    return parser.parse_args()


def test_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='zhan1993/flan-10k-flat-cluster-embedding')
    parser.add_argument("--project_name", type=str, default='Modular-LLM')
    parser.add_argument("--cluster_idx", type=int, required=True)  # The index of cluster that we want to train
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()
 
