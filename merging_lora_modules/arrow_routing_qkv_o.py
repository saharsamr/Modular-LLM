import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merging_lora_modules.base_merging_module import BaseMergingModule
import torch
from utils.config import cluster_checkpoint_names


class ArrowRouting(BaseMergingModule):
    def __init__(self, base_model, tokenizer, model_name, load_lora_modules=True):
        super().__init__(base_model, tokenizer, model_name)

        if load_lora_modules:
            self.load_lora_modules()
        

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

        # diff_AB = (U_W.T @ U_A).abs().diag()
        # if diff_AB[0] < 0.9:
        #     print("The first singular vector of U_A and U_AB are not aligned")

        return U_W, Sigma_C, V_W_T
    
    def routing_function(self):
        from local_peft.tuners.lora.layer import LoraLayer
        """
        This is the function responsible for computing prototype of the experts, after all the experts are loaded.
        """
        
        expert_counter = 0
    
        # As we iterate, each layer of model include 4 LoRA layer, q_proj, k_proj, v_proj, dense
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoraLayer): 
                print(name)
                # ** module.lora_A is a dict
                if expert_counter % 4 == 0:
                    q_proj = module
                elif expert_counter % 4 == 1:
                    k_proj = module
                elif expert_counter % 4 == 2: # We start to compute the qkv prototype
                    v_proj = module
                    for adapter_name in cluster_checkpoint_names.keys():
                        # Each lora_a weight shape: [8, 2560]
                        # concat_expert_A shape: [24, 2560]
                        # concat_expert_B shape: [2560, 24]
                        concat_expert_A = torch.cat((q_proj.lora_A[adapter_name].weight, k_proj.lora_A[adapter_name].weight, v_proj.lora_A[adapter_name].weight), dim=0)
                        concat_expert_B = torch.cat((q_proj.lora_B[adapter_name].weight, k_proj.lora_B[adapter_name].weight, v_proj.lora_B[adapter_name].weight), dim=1)
                        
                        A = concat_expert_A.T
                        B = concat_expert_B.T
                        
                        # Finding the prototypes in each layer (a.k.a LoRA layer, which is 2 in each model's layer)
                        W = (A @ B).T  # out_features, in_features

                        U_W, Sigma_W, V_W = self._low_rank_svd(A, B)
                        top_value = Sigma_W[0] ** 2
                        first_right_singular_vector = V_W.T[:, 0]
                        top_vector = U_W[:, 0]
                        # print(first_right_singular_vector.shape)
                        # print(top_vector.shape)

                        # Check that top vector is indeed an eigenvector
                        WTW = W.T @ W
                        ratio = WTW @ top_vector / (top_vector * top_value)
                        torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

                        # Check that top vector is indeed the top eigenvector
                        # assert (WTW @ top_vector).pow(2).sum() > (WTW @ bottom_vector).pow(
                        #     2
                        # ).sum()
                        
                        q_proj.experts_prototype[adapter_name] = first_right_singular_vector.detach()
                        k_proj.experts_prototype[adapter_name] = first_right_singular_vector.detach()
                        v_proj.experts_prototype[adapter_name] = first_right_singular_vector.detach()
                    
                elif expert_counter % 4 == 3: # We start to compute the dense layer prototype
                    dense = module
                    for adapter_name in cluster_checkpoint_names.keys():
                        # lora_a weight shape: [8, 2560]
                        # lora_b weight shape: [2560, 8]
                        
                        A = dense.lora_A[adapter_name].weight.T
                        B = dense.lora_B[adapter_name].weight.T
                        
                        # Finding the prototypes in each layer (a.k.a LoRA layer, which is 2 in each model's layer)
                        W = (A @ B).T  # out_features, in_features

                        U_W, Sigma_W, V_W = self._low_rank_svd(A, B)
                        top_value = Sigma_W[0] ** 2
                        first_right_singular_vector = V_W.T[:, 0]
                        top_vector = U_W[:, 0]
                        # print(first_right_singular_vector.shape)
                        # print(top_vector.shape)

                        # Check that top vector is indeed an eigenvector
                        WTW = W.T @ W
                        ratio = WTW @ top_vector / (top_vector * top_value)
                        torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

                        # Check that top vector is indeed the top eigenvector
                        # assert (WTW @ top_vector).pow(2).sum() > (WTW @ bottom_vector).pow(
                        #     2
                        # ).sum()
                        
                        dense.experts_prototype[adapter_name] = first_right_singular_vector.detach()
                
                expert_counter += 1

        print("The experts prototype have been computed successfully.")


    def merge(self, k):
        """
        This function completely does the merging
        """
        # Computing prototypes
        experts_prototypes, eigvals_dict = self.routing_function()


def _low_rank_svd(A, B):
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
    # if diff_AB[0] < 0.9:
    #     print("The first singular vector of U_A and U_AB are not aligned")

    return U_W, Sigma_C, V_W_T


def compute_weight(current_input, experts_prototypes, top_k):
    """
    This function computes the coefficients for each expert in each layer.
    """

    # Assuming the following variables are defined:
    # experts_prototypes: dict of expert_name -> tensor of shape [model_dim]
    # current_input: tensor of shape [batch_size, token_num, model_dim]

    # Step 1: Stack all expert prototypes into a tensor
    # Shape: [num_experts, model_dim]
    experts_matrix = torch.stack([experts_prototypes[name] for name in experts_prototypes.keys()], dim=0)

    # Ensure current_input is of type float32
    current_input_float = current_input.to(torch.float32)

    # Step 2: Perform batch matrix multiplication
    # current_input_float: [batch_size, token_num, model_dim]
    # experts_matrix: [model_dim, num_experts]
    # Resulting logits: [batch_size, token_num, num_experts]
    logits = torch.matmul(current_input_float, experts_matrix.T)

    # Step 3: Apply absolute value and permute dimensions
    # logits: [batch_size, token_num, num_experts]
    logits_mat = torch.abs(logits)
    
    # Get the top k values and their indices along the last dim
    top_k_values, top_k_indices = torch.topk(logits_mat, top_k, dim=-1)

    # Create an output matrix filled with -inf
    output_matrix = torch.full_like(logits_mat, -float('inf'))

    # Scatter the top k values into their respective positions in the output matrix
    output_matrix.scatter_(-1, top_k_indices, top_k_values)

    # Apply softmax
    softmax = torch.nn.Softmax(dim=-1)

    return softmax(output_matrix)



