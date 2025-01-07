import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from peft import PeftModel
from local_peft import PeftModel
from transformers import AutoTokenizer, AutoConfig

from utils.config import EXPERTS_FOLDER_PATH, MAX_LENGTH


cluster_checkpoint_names = {
    'cluster0': f'{EXPERTS_FOLDER_PATH}/cluster0_batch2_prop0.2/checkpoint-2507/',
    'cluster1': f'{EXPERTS_FOLDER_PATH}/cluster1_batch2_prop0.2/checkpoint-2506/',
    'cluster2': f'{EXPERTS_FOLDER_PATH}/cluster2_batch2_prop0.2/checkpoint-2440/',
    'cluster3': f'{EXPERTS_FOLDER_PATH}/cluster3_batch2_prop0.2/checkpoint-2397/',
    'cluster4': f'{EXPERTS_FOLDER_PATH}/cluster4_batch2_prop0.2/checkpoint-2383/',
    'cluster5': f'{EXPERTS_FOLDER_PATH}/cluster5_batch2_prop0.2/checkpoint-2372/',
    'cluster6': f'{EXPERTS_FOLDER_PATH}/cluster6_batch2_prop0.2/checkpoint-2362/',
    'cluster7': f'{EXPERTS_FOLDER_PATH}/cluster7_batch2_prop0.2/checkpoint-2356/',
    'cluster8': f'{EXPERTS_FOLDER_PATH}/cluster8_batch2_prop0.2/checkpoint-2320/',
    'cluster9': f'{EXPERTS_FOLDER_PATH}/cluster9_batch2_prop0.2/checkpoint-2053/',
}


class BaseMergingModule:
    def __init__(self, base_model, tokenizer, model_name):
        self.base_model = base_model
        self.base_model.enable_input_require_grads()
        self.model_name = model_name
        self.base_model_config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_flash_attention_2=False,
            device_map="auto",
        )
        self.max_length = MAX_LENGTH
        self.tokenizer = tokenizer

    def merge(self):
        raise NotImplementedError

    def load_lora_modules(self):
        self.base_model = PeftModel.from_pretrained(
            self.base_model, cluster_checkpoint_names['cluster0'], adapter_name='cluster0')
        for cluster_name, cluster_path in cluster_checkpoint_names.items():
            if cluster_name != 'cluster0':
                self.base_model.load_adapter(cluster_path, adapter_name=cluster_name)

    def get_model(self):
        return self.base_model
