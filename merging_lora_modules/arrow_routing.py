from merging_lora_modules.base_merging_module import BaseMergingModule, cluster_checkpoint_names
from peft.tuners.lora.layer import LoraLayer
import torch
from tensordict.tensordict import TensorDict
from torch import nn


class ArrowRouting(BaseMergingModule):
    def __init__(self, base_model, tokenizer, model_name):
        super().__init__(base_model, tokenizer, model_name)
        

    def _low_rank_svd(self, A, B):
        """Faster SVD computation for low rank matrices"""

        # Compute SVD of A
        U_A, Sigma_A, V_A = torch.svd(A)

        # Compute SVD of B.T (transpose of B)
        U_B, Sigma_B, V_B = torch.svd(B.T)

        # Compute product matrix C = Sigma_A * (V_A.T @ V_B) * Sigma_B
        # Since V_A and V_B are orthogonal, their product is also an orthogonal matrix
        C = Sigma_A.diag_embed() @ V_A.t() @ V_B @ Sigma_B.diag_embed()

        # Compute SVD of the product matrix C
        U_C, Sigma_C, V_C = torch.svd(C)

        # Construct the final SVD components of W
        U_W = U_A @ U_C
        V_W_T = V_C.t() @ U_B.t()

        diff_AB = (U_W.T @ U_A).abs().diag()
        if diff_AB[0] < 0.9:
            print("The first singular vector of U_A and U_AB are not aligned")

        return U_W, Sigma_C, V_W_T
    

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

        vectors_dict = {i : {j : 0 for j in all_lora_dict[i]['lora_A'].keys()} for i in all_lora_dict.keys()}
        eigvals_dict = {i : {j : 0 for j in all_lora_dict[i]['lora_A'].keys()} for i in all_lora_dict.keys()}

        for layer in all_lora_dict.keys():
            for cluster in all_lora_dict[layer]['lora_A'].keys():

                A = all_lora_dict[layer]['lora_A'][cluster]
                # A = [tensor.tolist() for tensor in A.values()]

                B = all_lora_dict[layer]['lora_B'][cluster]
                # B = [tensor.tolist() for tensor in B.values()]

                # print(torch.tensor(A).size())
                # print(torch.tensor(B).size())

                # A = torch.cat(torch.tensor(A), dim=1)
                # B = torch.cat(torch.tensor(B), dim=0)

                # rank = 4

                # A = A.reshape(-1, rank).float()
                # B = B.reshape(rank, -1).float()

                W = (A @ B).T  # out_features, in_features

                U_W, Sigma_W, _ = self._low_rank_svd(A, B)
                top_value = Sigma_W[0] ** 2
                bottom_vector = U_W[:, -1]
                top_vector = U_W[:, 0]

                # Check that top vector is indeed an eigenvector
                WTW = W.T @ W
                ratio = WTW @ top_vector / (top_vector * top_value)
                torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

                # Check that top vector is indeed the top eigenvector
                assert (WTW @ top_vector).pow(2).sum() > (WTW @ bottom_vector).pow(
                    2
                ).sum()

                # Save eigenvector and eigvenvalue
                vectors_dict[layer][cluster] = top_vector.detach().cpu().numpy()
                eigvals_dict[layer][cluster] = top_value.item()

        print(vectors_dict, eigvals_dict)

        return vectors_dict, eigvals_dict

    def get_model(self, vectors_dict):
        pass
