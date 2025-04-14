# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from local_peft.tuners.lora.layer import LoraLayer
from merging_lora_modules.base_merging_module import BaseMergingModule, cluster_checkpoint_names
from peft import PeftModel
import torch


class CrossLingualExpertOrganiser(BaseMergingModule):
    def __init__(
            self, base_model, tokenizer, model_name,
            source_formal_expert_path, target_formal_expert_path,
            method, alpha=1, beta=1, load_single_expert=False, cluster_name=None
    ):
        super().__init__(base_model, tokenizer, model_name)

        self.source_formal_expert_path = source_formal_expert_path
        self.target_formal_expert_path = target_formal_expert_path
        self.method = method
        # a and b are coefficients for functional and target formal expert for being combined
        self.alpha = alpha
        self.beta = beta

        if load_single_expert:
            assert cluster_name, 'Cluster idx should be passed'
            self.base_model = PeftModel.from_pretrained(
                self.base_model, cluster_checkpoint_names[cluster_name], adapter_name=cluster_name)
            self.cluster_names = {cluster_name: cluster_checkpoint_names[cluster_name]}
        else:
            self.load_lora_modules()
            self.cluster_names = cluster_checkpoint_names

    

    def create_functional_modules(self, use_avg_lora):
        self.base_model.load_adapter(self.source_formal_expert_path, adapter_name='source_formal_expert')
        
        if self.method == 'subtract':
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    for adapter_name in self.cluster_names.keys():
                        if use_avg_lora:
                            module.lora_A[adapter_name].weight = torch.nn.Parameter(
                                module.lora_A[adapter_name].weight - (1/1)*module.lora_A['average'].weight)
                            module.lora_B[adapter_name].weight = torch.nn.Parameter(
                                module.lora_B[adapter_name].weight - (1/1)*module.lora_B['average'].weight)
                        else:
                            module.lora_A[adapter_name].weight = torch.nn.Parameter(
                                module.lora_A[adapter_name].weight - (1/1)*module.lora_A['source_formal_expert'].weight)
                            module.lora_B[adapter_name].weight = torch.nn.Parameter(
                                module.lora_B[adapter_name].weight - (1/1)*module.lora_B['source_formal_expert'].weight)

        elif self.method == 'orthogonal_projection':
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    if use_avg_lora:
                        Q_A, _ = torch.linalg.qr(module.lora_A['average'].weight.T)
                        Q_B, _ = torch.linalg.qr(module.lora_B['average'].weight)
                    else:
                        Q_A, _ = torch.linalg.qr(module.lora_A['source_formal_expert'].weight.T)
                        Q_B, _ = torch.linalg.qr(module.lora_B['source_formal_expert'].weight)

                    for adapter_name in self.cluster_names.keys():
                        mixed_expert_A = module.lora_A[adapter_name].weight.T
                        mixed_expert_B = module.lora_B[adapter_name].weight

                        projection_coefficients_A = Q_A.T @ mixed_expert_A  
                        projection_coefficients_B = Q_B.T @ mixed_expert_B  

                        mixed_project_on_formal_A = Q_A @ projection_coefficients_A  
                        mixed_project_on_formal_B = Q_B @ projection_coefficients_B

                        module.lora_A[adapter_name].weight = torch.nn.Parameter(mixed_expert_A.T - mixed_project_on_formal_A.T)
                        module.lora_B[adapter_name].weight = torch.nn.Parameter(mixed_expert_B - mixed_project_on_formal_B)

        self.base_model.delete_adapter("source_formal_expert")
        print("Functional Expert has been created successfully.")

    def merge(self, add_functional_only, use_avg_lora=False):
        if use_avg_lora:
            self.average_lora_modules()

        if add_functional_only:
            # TODO: add 3 level if elif else
            # self.create_functional_modules(use_avg_lora) # Isolating from source lang
            self.source_formal_expert_path = self.target_formal_expert_path
            self.create_functional_modules(use_avg_lora) # Isolating from target lang
            # self.source_formal_expert_path = "/home/tmptildec/Ali/Modular-LLM-LE/scripts/results/cluster0_batch16_prop1.0_langger/checkpoint-17"
            # self.create_functional_modules(use_avg_lora) 
            # self.base_model.load_adapter(self.target_formal_expert_path, adapter_name='target_formal_expert') # Adding target language as the 11th adapter
            # cluster_checkpoint_names["target_formal_expert"] = self.target_formal_expert_path
            # print(cluster_checkpoint_names)
        else:
            self.create_functional_modules(use_avg_lora)
            self.create_cross_lingual_expert()

        # if use_avg_lora:
        #     self.base_model.delete_adapter('average')
