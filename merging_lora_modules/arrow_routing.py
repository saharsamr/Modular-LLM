from merging_lora_modules.base_merging_module import BaseMergingModule, cluster_checkpoint_names
from peft.tuners.lora.layer import LoraLayer
import torch
from tensordict.tensordict import TensorDict
from torch import nn


class ArrowRouting(BaseMergingModule):
    def __init__(self, base_model, tokenizer, model_name):
        super().__init__(base_model, tokenizer, model_name)
    

    def routing_function(self):
        # We first load the lora modules
        self.load_lora_modules()
        # # Then we iterate on the adapters on the base model.
        # print(self.base_model)
        # print('='*50)
        
        proj_counter = 0
        all_lora_dict = {}
        # As we iterate, each layer of model include 2 LoRA layer, o_proj [(3072, 4), (4, 3072)] and 
        # qkv_proj [(3072, 4), (4, 9216)]
        for module in self.base_model.modules(): 
            if isinstance(module, LoraLayer): 
                if proj_counter % 2 == 0:
                    # module is o_proj
                    o_proj_lora_A_module_dict = module.lora_A
                    o_proj_lora_B_module_dict = module.lora_B
                else:
                    # module is qkv_proj
                    qkv_proj_lora_A_module_dict = module.lora_A
                    qkv_proj_lora_B_module_dict = module.lora_B

                    # Now we concat o_proj and qkv_proj on dim = 1
                    concat_lora_A_dict = TensorDict()
                    concat_lora_B_dict = TensorDict()

                    for adapter_name in cluster_checkpoint_names.keys():
                        o_proj_lora_A = o_proj_lora_A_module_dict[adapter_name].weight.T
                        o_proj_lora_B = o_proj_lora_B_module_dict[adapter_name].weight

                        qkv_proj_lora_A = qkv_proj_lora_A_module_dict[adapter_name].weight.T
                        qkv_proj_lora_B = qkv_proj_lora_B_module_dict[adapter_name].weight

                        expert_lora_A = torch.cat((o_proj_lora_A, qkv_proj_lora_A), dim=0)
                        expert_lora_B = torch.cat((o_proj_lora_B, qkv_proj_lora_B), dim=0).T

                        concat_lora_A_dict[adapter_name] = expert_lora_A
                        concat_lora_B_dict[adapter_name] = expert_lora_B
                    
                    # Now we add both lora_A and lora_B dictionaries to the global dictionary
                    all_lora_dict['layer' + str(proj_counter // 2)] = {'lora_A': concat_lora_A_dict, 'lora_B': concat_lora_B_dict}

                proj_counter += 1
        
        print(all_lora_dict)
        
        # TODO: Computing prototype using SVD and then finding coefficients.

                
        


