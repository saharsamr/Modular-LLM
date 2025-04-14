import sys
import os
import torch

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from peft import PeftModel
from local_peft import PeftModel
from transformers import AutoConfig

from utils.config import MAX_LENGTH, cluster_checkpoint_names 
    
# cluster_checkpoint_names = {
#     'cluster0': 'scripts/results/phi2/cluster0_batch1_prop1.0/checkpoint-2000/',
#     'cluster1': 'scripts/results/phi2/cluster1_batch1_prop1.0/checkpoint-2000/',
#     'cluster2': 'scripts/results/phi2/cluster2_batch1_prop1.0/checkpoint-2000/',
#     'cluster3': 'scripts/results/phi2/cluster3_batch1_prop1.0/checkpoint-2000/',
#     'cluster4': 'scripts/results/phi2/cluster4_batch1_prop1.0/checkpoint-2000/',
#     'cluster5': 'scripts/results/phi2/cluster5_batch1_prop1.0/checkpoint-2372/',
#     'cluster6': f'{EXPERTS_FOLDER_PATH}/cluster6_batch1_prop1.0/checkpoint-2362/',
#     'cluster7': f'{EXPERTS_FOLDER_PATH}/cluster7_batch1_prop1.0/checkpoint-2356/',
#     'cluster8': f'{EXPERTS_FOLDER_PATH}/cluster8_batch1_prop1.0/checkpoint-2320/',
#     'cluster9': f'{EXPERTS_FOLDER_PATH}/cluster9_batch1_prop1.0/checkpoint-2053/',
# }


class BaseMergingModule:
    def __init__(self, base_model, tokenizer, model_name):
        self.base_model = base_model
        self.base_model_no_peft = base_model
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
    

    def check_model_weights_difference(self):
        # Checking if the model weights after loading LoRAs are as same as before.
        # Check only the base model parameters (excluding LoRA parameters)
        print(self.base_model_no_peft)
        print("-"*50)
        print(self.base_model)
        for name, base_param in self.base_model_no_peft.named_parameters():
            if name in self.base_model.state_dict():  # Ensure parameter exists in the LoRA model
                lora_param = self.base_model.state_dict()[name]
                if not torch.equal(base_param.data, lora_param.data):
                    print(f"Mismatch found in base model parameter: {name}")
                    return False
            else:
                print(f"Parameter {name} from base model not found in LoRA-applied model.")
                return False

        print("All base model weights are unchanged.")
        return True
