# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.arg_parser import cross_lingual_arg_parser
from utils.config import EXPERTS_FOLDER_PATH, MAX_LENGTH
from local_peft import PeftModel
from local_peft.tuners.lora.layer import LoraLayer
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CLUSTER_CHECKPOINT_NAMES = {
    'cluster0': f'{EXPERTS_FOLDER_PATH}/cluster0_batch1_prop0.2/checkpoint-2507/',
    'cluster1': f'{EXPERTS_FOLDER_PATH}/cluster1_batch1_prop0.2/checkpoint-2506/',
    'cluster2': f'{EXPERTS_FOLDER_PATH}/cluster2_batch1_prop0.2/checkpoint-2440/',
    'cluster3': f'{EXPERTS_FOLDER_PATH}/cluster3_batch1_prop0.2/checkpoint-2397/',
    'cluster4': f'{EXPERTS_FOLDER_PATH}/cluster4_batch1_prop0.2/checkpoint-2383/',
    'cluster5': f'{EXPERTS_FOLDER_PATH}/cluster5_batch1_prop0.2/checkpoint-2372/',
    'cluster6': f'{EXPERTS_FOLDER_PATH}/cluster6_batch1_prop0.2/checkpoint-2362/',
    'cluster7': f'{EXPERTS_FOLDER_PATH}/cluster7_batch1_prop0.2/checkpoint-2356/',
    'cluster8': f'{EXPERTS_FOLDER_PATH}/cluster8_batch1_prop0.2/checkpoint-2320/',
    'cluster9': f'{EXPERTS_FOLDER_PATH}/cluster9_batch1_prop0.2/checkpoint-2053/',
}


class CrossLingualExpertOrganiser:
    def __init__(self, base_model, tokenizer, source_formal_expert_path, target_formal_expert_path, method, alpha=0.5, beta=0.5):
        self.base_model = base_model
        self.tokenizer = tokenizer
        # The paths to source_formal_expert and target_formal_expert
        self.source_formal_expert_path = source_formal_expert_path
        self.target_formal_expert_path = target_formal_expert_path
        self.method = method
        # a and b are coefficients for functional and target formal expert for being combined
        self.alpha = alpha
        self.beta = beta

    def load_mixed_lora_modules(self):
        self.base_model = PeftModel.from_pretrained(
            self.base_model, CLUSTER_CHECKPOINT_NAMES['cluster0'], adapter_name='cluster0')
        for cluster_name, cluster_path in CLUSTER_CHECKPOINT_NAMES.items():
            if cluster_name != 'cluster0':
                self.base_model.load_adapter(cluster_path, adapter_name=cluster_name)

    def create_functional_modules(self):
        # We first load the source_formal_expert on the model (we'll have 11 experts)
        self.base_model.load_adapter(self.source_formal_expert_path, adapter_name='source_formal_expert')
        
        if self.method == 'subtract':
            module_idx = 0
            # Iterating over the modules
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    for adapter_name in CLUSTER_CHECKPOINT_NAMES.keys():
                        former_weight_A = module.lora_A[adapter_name].weight
                        former_weight_B = module.lora_B[adapter_name].weight

                        module.lora_A[adapter_name].weight = torch.nn.Parameter(module.lora_A[adapter_name].weight - module.lora_A['source_formal_expert'].weight)
                        module.lora_B[adapter_name].weight = torch.nn.Parameter(module.lora_B[adapter_name].weight - module.lora_B['source_formal_expert'].weight)

                        # delta_A = module.lora_A[adapter_name].weight - former_weight_A
                        # delta_B = module.lora_B[adapter_name].weight - former_weight_B

                        # if module_idx % 2 == 0:
                        #     print(f"Block {module_idx // 2} - {adapter_name} - o_proj")
                        # elif module_idx % 2 == 1:
                        #     print(f"Block {module_idx // 2} - {adapter_name} - qkv_proj")
                        # # Verification of changes of weight
                        # print("Frobenius Norm (Overall Difference) of module A:", torch.norm(delta_A, p="fro").item())
                        # print("Frobenius Norm (Overall Difference) of module B:", torch.norm(delta_B, p="fro").item())
                        # print("Former weight of module A:", former_weight_A)
                        # print("Former weight of module B:", former_weight_B)
                        # print("="*100)

                    module_idx += 1


        
        elif self.method == 'orthogonal_projection':
            module_idx = 0
            # Iterating over the modules
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    # QR Decomposition on formal expert in the layer
                    Q_A, _ = torch.linalg.qr(module.lora_A['source_formal_expert'].weight.T) 
                    # Q_A shape: (3072, 4)
                    Q_B, _ = torch.linalg.qr(module.lora_B['source_formal_expert'].weight) 
                    # Q_B shape: (3072, 4)

                    for adapter_name in CLUSTER_CHECKPOINT_NAMES.keys():
                        mixed_expert_A = module.lora_A[adapter_name].weight.T
                        mixed_expert_B = module.lora_B[adapter_name].weight

                        projection_coefficients_A = Q_A.T @ mixed_expert_A  
                        projection_coefficients_B = Q_B.T @ mixed_expert_B  

                        mixed_project_on_formal_A = Q_A @ projection_coefficients_A  
                        mixed_project_on_formal_B = Q_B @ projection_coefficients_B  

                        # Getting the isolated functional component
                        former_weight_A = module.lora_A[adapter_name].weight
                        former_weight_B = module.lora_B[adapter_name].weight

                        module.lora_A[adapter_name].weight = torch.nn.Parameter(mixed_expert_A.T - mixed_project_on_formal_A.T)
                        module.lora_B[adapter_name].weight = torch.nn.Parameter(mixed_expert_B - mixed_project_on_formal_B)

                        # delta_A = module.lora_A[adapter_name].weight - former_weight_A
                        # delta_B = module.lora_B[adapter_name].weight - former_weight_B

                        # if module_idx % 2 == 0:
                        #     print(f"Block {module_idx // 2} - {adapter_name} - o_proj")
                        # elif module_idx % 2 == 1:
                        #     print(f"Block {module_idx // 2} - {adapter_name} - qkv_proj")
                        # # Verification of changes of weight
                        # print("Frobenius Norm (Overall Difference) of module A:", torch.norm(delta_A, p="fro").item())
                        # print("Frobenius Norm (Overall Difference) of module B:", torch.norm(delta_B, p="fro").item())
                        # print("Former weight of module A:", former_weight_A)
                        # print("Former weight of module B:", former_weight_B)
                        # print("="*100)

                        # print(f"The weight's change in module A is: {module.lora_A[adapter_name].weight - former_weight_A}")
                        # print(f"The weight's change in module B is: {module.lora_B[adapter_name].weight - former_weight_B}")
                        # # Verification of isolation
                        # dot_product_A = torch.dot(module.lora_A[adapter_name].weight.flatten(), mixed_project_on_formal_A.T.flatten())
                        # print(f"Dot product for A (should be near 0): {dot_product_A}")
                        # dot_product_B = torch.dot(module.lora_B[adapter_name].weight.flatten(), mixed_project_on_formal_B.flatten())
                        # print(f"Dot product for B (should be near 0): {dot_product_B}")

                    module_idx += 1
        
        # We are done with source_formal_expert, so we delete it.
        self.base_model.delete_adapter("source_formal_expert")
    
    def create_cross_lingual_expert(self):
        # We first load the target_formal_expert on the model (we'll have 11 experts)
        self.base_model.load_adapter(self.target_formal_expert_path, adapter_name='target_formal_expert')

        # Iterating over the modules
        for module in self.base_model.modules():
            if isinstance(module, LoraLayer):
                for adapter_name in CLUSTER_CHECKPOINT_NAMES.keys():
                    module.lora_A[adapter_name].weight = torch.nn.Parameter(self.alpha * module.lora_A[adapter_name].weight + self.beta * module.lora_A['target_formal_expert'].weight)
                    module.lora_B[adapter_name].weight = torch.nn.Parameter(self.alpha * module.lora_B[adapter_name].weight + self.beta * module.lora_B['target_formal_expert'].weight)

        # We are done with source_formal_expert, so we delete it.
        self.base_model.delete_adapter("target_formal_expert")
    
    def get_model(self):
        return self.base_model


if __name__ == "__main__":

    args = cross_lingual_arg_parser()
    set_seed(args.seed)

    # Loading the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, padding_side='right', model_max_length=MAX_LENGTH
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print('Loading Model ...')
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, quantization_config=bnb_config)
    
    # Now we initialise the cross lingual expert
    cle_org = CrossLingualExpertOrganiser(model, 
                                   tokenizer, 
                                   args.source_formal_expert_path, 
                                   args.target_formal_expert_path,
                                   args.method
                                   )

    # Loading 10 cluster's expert on the base model
    cle_org.load_mixed_lora_modules()

    # Creating functional expert
    cle_org.create_functional_modules()

    # Creating cross lingual experts
    cle_org.create_cross_lingual_expert()

    # Get the base model after building cross_lingual expert
    cross_lingual_model = cle_org.get_model()

    print(cross_lingual_model)