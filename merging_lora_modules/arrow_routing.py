import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merging_lora_modules.base_merging_module import BaseMergingModule, cluster_checkpoint_names
# from peft.tuners.lora.layer import LoraLayer
from local_peft.tuners.lora.layer import LoraLayer
import torch
from tensordict.tensordict import TensorDict
from torch import nn
import numpy as np

from transformers import GenerationConfig

class CustomModel(nn.Module):
    def __init__(self, base_model, base_model_config, experts_prototypes, k=1):
        super().__init__()
        # # Load the base model with the first cluster (default LoRA adapter)
        # self.base_model = PeftModel.from_pretrained(
        #     base_model, cluster_checkpoint_names['cluster0'], adapter_name='cluster0')
        
        # # Load additional experts (LoRA adapters)
        # for cluster_name, cluster_path in cluster_checkpoint_names.items():
        #     if cluster_name != 'cluster0':
        #         self.base_model.load_adapter(cluster_path, adapter_name=cluster_name)
        
        # base_model is already loaded with adapters.
        self.base_model = base_model
        # Store the expert mapping for each layer
        self.experts_prototypes = experts_prototypes
        # k_best expert that we wanna use
        self.k = k
        # Necessary attributes for pipeline()
        self.config = base_model_config

        # Adding generation_config attribute
        self.generation_config = GenerationConfig.from_pretrained(base_model_config._name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def can_generate(self):
        return True

    def expert_mapping(self, layer_index, current_input):
        """
        This function computes the coefficients for each expert in each layer.
        """
        # Computing logits
        logits = {}
        for expert_name in self.experts_prototypes[layer_index].keys():
            logits[expert_name] = torch.abs(torch.dot(self.experts_prototypes[layer_index][expert_name], current_input))

        # Sort the logits based on the values and return a dict
        sorted_logits = dict(sorted(logits.items(), key=lambda item: item[1], reverse=True))

        # Select top_k and set others as -infinity
        for i, (k, v) in enumerate(sorted_logits.items()):
            if i < self.k:
                sorted_logits[k] = v
            else:
                sorted_logits[k] = -np.inf

        # Applying softmax
        def softmax_on_dict(logits_dict):
            x = np.fromiter(logits_dict.values(), dtype=float)
            softmax_scores = np.exp(x) / np.sum(np.exp(x), axis=0)
            return logits_dict.update(zip(logits_dict, softmax_scores))

        return softmax_on_dict(sorted_logits)
        

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Initial input is the input_ids from the text
        current_input = input_ids
        
        # Iterate through each layer and apply the corresponding expert
        for layer_index, layer in enumerate(self.base_model.model.layers):
            # Computing the weights for the corresponding layer
            weights = self.expert_mapping(layer_index, current_input)

            print(weights)
            print('='*30)
            # TODO: Multiplying weights with experts
            print(layer) # To see what's the structure of the layer

            # Switch to the specific LoRA expert for this layer
            # self.base_model.set_adapter(f'cluster{expert_index}')

            # Perform the forward pass for this layer using the current input
            output = layer(current_input, attention_mask=attention_mask, token_type_ids=token_type_ids)

            # Update the input for the next layer to be the output of the current layer
            current_input = output

        # Return the final output after passing through all layers
        return current_input

    def generate(self, input_ids, attention_mask, pad_token_id='', eos_token_id='', max_new_tokens=50, max_length=50, num_beams=5, do_sample=True, top_p=0.9, temperature=0.8, num_return_sequences=3):
      return input_ids


# class CustomPeftModel(PeftModel):
#     def __init__(self, base_model, peft_config):
#         super().__init__(base_model, peft_config)

#     def set_layer_expert():
#         pass

#     def forward(self, *args, **kwargs):
#         # Iterate through each layer and apply the corresponding expert
#         for layer_index, layer in enumerate(self.base_model.model.layers):
#             # Select the expert based on the mapping for the current layer
#             expert_index = self.expert_mapping[layer_index]

#             # Switch to the specific LoRA expert for this layer
#             self.base_model.set_adapter(f'cluster{expert_index}')

#             # Perform the forward pass for this layer
#             output = layer(*args, **kwargs)
            
#             # Process the output (you may need to modify this based on your model structure)
#             # Example: you may want to store or process the output here
        
#         # Continue the rest of the forward pass as usual
#         return output


class ArrowRouting(BaseMergingModule):
    def __init__(self, base_model, tokenizer, model_name):
        super().__init__(base_model, tokenizer, model_name)

    # def load_lora_modules(self):
    #     self.base_model = CustomPeftModel.from_pretrained(
    #         self.base_model, cluster_checkpoint_names['cluster0'], adapter_name='cluster0')
    #     for cluster_name, cluster_path in cluster_checkpoint_names.items():
    #         if cluster_name != 'cluster0':
    #             self.base_model.load_adapter(cluster_path, adapter_name=cluster_name)
        

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
        """
        This is the function responsible for computing prototype of the experts.
        The prototypes will be passed to the CustomModel class as the argument to the cunstructor
        """

        # We first load the lora modules
        self.load_lora_modules()
        
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
        
        # print(all_lora_dict)

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

        # print(vectors_dict, eigvals_dict)

        return vectors_dict, eigvals_dict

    def merge(self, k):
        """
        This function completely does the merging and return the model with new merged adapters
        """
        # Computing prototypes
        experts_prototypes, eigvals_dict = self.routing_function()

        # Creating CustomModel
        self.base_model = CustomModel(self.base_model, self.base_model_config, experts_prototypes, k)





