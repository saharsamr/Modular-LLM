from merging_lora_modules.base_merging_module import cluster_checkpoint_names
from huggingface_hub import HfApi


if __name__ == "__main__":
    api = HfApi()

    for model_ckpt_name, model_ckpt_path in cluster_checkpoint_names.items():
        api.upload_folder(
            folder_path=model_ckpt_path,
            repo_id="AliEdalat/experts",
            path_in_repo=model_ckpt_name,
            token='ACCESS_TOKEN',
        )