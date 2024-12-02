# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from local_peft.tuners.lora.layer import LoraLayer
from merging_lora_modules.base_merging_module import BaseMergingModule, cluster_checkpoint_names
import torch


class CrossLingualExpertOrganiser(BaseMergingModule):
    def __init__(self, base_model, tokenizer, model_name, source_formal_expert_path, target_formal_expert_path, method, alpha=0.5, beta=0.5):
        super().__init__(base_model, tokenizer, model_name)

        self.source_formal_expert_path = source_formal_expert_path
        self.target_formal_expert_path = target_formal_expert_path
        self.method = method
        # a and b are coefficients for functional and target formal expert for being combined
        self.alpha = alpha
        self.beta = beta

        self.load_lora_modules()

    def create_functional_modules(self):
        self.base_model.load_adapter(self.source_formal_expert_path, adapter_name='source_formal_expert')
        
        if self.method == 'subtract':
            module_idx = 0
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    for adapter_name in cluster_checkpoint_names.keys():
                        module.lora_A[adapter_name].weight = torch.nn.Parameter(module.lora_A[adapter_name].weight - module.lora_A['source_formal_expert'].weight)
                        module.lora_B[adapter_name].weight = torch.nn.Parameter(module.lora_B[adapter_name].weight - module.lora_B['source_formal_expert'].weight)

                    module_idx += 1

        elif self.method == 'orthogonal_projection':
            module_idx = 0
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    # QR Decomposition on formal expert in the layer
                    Q_A, _ = torch.linalg.qr(module.lora_A['source_formal_expert'].weight.T)
                    Q_B, _ = torch.linalg.qr(module.lora_B['source_formal_expert'].weight)

                    for adapter_name in cluster_checkpoint_names.keys():
                        mixed_expert_A = module.lora_A[adapter_name].weight.T
                        mixed_expert_B = module.lora_B[adapter_name].weight

                        projection_coefficients_A = Q_A.T @ mixed_expert_A  
                        projection_coefficients_B = Q_B.T @ mixed_expert_B  

                        mixed_project_on_formal_A = Q_A @ projection_coefficients_A  
                        mixed_project_on_formal_B = Q_B @ projection_coefficients_B

                        module.lora_A[adapter_name].weight = torch.nn.Parameter(mixed_expert_A.T - mixed_project_on_formal_A.T)
                        module.lora_B[adapter_name].weight = torch.nn.Parameter(mixed_expert_B - mixed_project_on_formal_B)

                    module_idx += 1

        self.base_model.delete_adapter("source_formal_expert")
    
    def create_cross_lingual_expert(self):
        self.base_model.load_adapter(self.target_formal_expert_path, adapter_name='target_formal_expert')

        for module in self.base_model.modules():
            if isinstance(module, LoraLayer):
                for adapter_name in cluster_checkpoint_names.keys():
                    module.lora_A[adapter_name].weight = torch.nn.Parameter(self.alpha * module.lora_A[adapter_name].weight + self.beta * module.lora_A['target_formal_expert'].weight)
                    module.lora_B[adapter_name].weight = torch.nn.Parameter(self.alpha * module.lora_B[adapter_name].weight + self.beta * module.lora_B['target_formal_expert'].weight)

        self.base_model.delete_adapter("target_formal_expert")

    def merge(self, add_functional_only):
        if add_functional_only:
            self.create_functional_modules()
        else:
            self.create_functional_modules()
            self.create_cross_lingual_expert()
