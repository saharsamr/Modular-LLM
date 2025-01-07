import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merging_lora_modules.base_merging_module import BaseMergingModule
# from local_peft.tuners.lora.layer import LoraLayer
import torch


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

        diff_AB = (U_W.T @ U_A).abs().diag()
        if diff_AB[0] < 0.9:
            print("The first singular vector of U_A and U_AB are not aligned")

        return U_W, Sigma_C, V_W_T
    
    # def routing_function(self):
    #     """
    #     This is the function responsible for computing prototype of the experts.
    #     The prototypes will be passed to the CustomModel class as the argument to the cunstructor
    #     """

    #     # We first load the lora modules
    #     self.load_lora_modules()

    #     proj_counter = 0
    #     all_lora_dict = {}
    #     # As we iterate, each layer of model include 2 LoRA layer, o_proj [(3072, 4), (4, 3072)] and
    #     # qkv_proj [(3072, 4), (4, 9216)]
    #     for module in self.base_model.modules():
    #         if isinstance(module, LoraLayer):
    #             if proj_counter % 2 == 0:
    #                 # module is o_proj
    #                 o_proj_lora_A_module_dict = module.lora_A
    #                 o_proj_lora_B_module_dict = module.lora_B
    #             else:
    #                 # module is qkv_proj
    #                 qkv_proj_lora_A_module_dict = module.lora_A
    #                 qkv_proj_lora_B_module_dict = module.lora_B

    #                 # Now we concat o_proj and qkv_proj on dim = 1
    #                 concat_lora_A_dict = TensorDict()
    #                 concat_lora_B_dict = TensorDict()

    #                 for adapter_name in cluster_checkpoint_names.keys():
    #                     o_proj_lora_A = o_proj_lora_A_module_dict[adapter_name].weight.T
    #                     o_proj_lora_B = o_proj_lora_B_module_dict[adapter_name].weight

    #                     qkv_proj_lora_A = qkv_proj_lora_A_module_dict[adapter_name].weight.T
    #                     qkv_proj_lora_B = qkv_proj_lora_B_module_dict[adapter_name].weight

    #                     expert_lora_A = torch.cat((o_proj_lora_A, qkv_proj_lora_A), dim=0)
    #                     expert_lora_B = torch.cat((o_proj_lora_B, qkv_proj_lora_B), dim=0).T

    #                     concat_lora_A_dict[adapter_name] = expert_lora_A
    #                     concat_lora_B_dict[adapter_name] = expert_lora_B

    #                 # Now we add both lora_A and lora_B dictionaries to the global dictionary
    #                 all_lora_dict['layer' + str(proj_counter // 2)] = {'lora_A': concat_lora_A_dict, 'lora_B': concat_lora_B_dict}

    #             proj_counter += 1

    #     # print(all_lora_dict)

    #     vectors_dict = {i : {j : 0 for j in all_lora_dict[i]['lora_A'].keys()} for i in all_lora_dict.keys()}
    #     eigvals_dict = {i : {j : 0 for j in all_lora_dict[i]['lora_A'].keys()} for i in all_lora_dict.keys()}

    #     for layer in all_lora_dict.keys():
    #         for cluster in all_lora_dict[layer]['lora_A'].keys():

    #             A = all_lora_dict[layer]['lora_A'][cluster]
    #             # A = [tensor.tolist() for tensor in A.values()]

    #             B = all_lora_dict[layer]['lora_B'][cluster]
    #             # B = [tensor.tolist() for tensor in B.values()]

    #             # print(torch.tensor(A).size())
    #             # print(torch.tensor(B).size())

    #             # A = torch.cat(torch.tensor(A), dim=1)
    #             # B = torch.cat(torch.tensor(B), dim=0)

    #             # rank = 4

    #             # A = A.reshape(-1, rank).float()
    #             # B = B.reshape(rank, -1).float()

    #             W = (A @ B).T  # out_features, in_features

    #             U_W, Sigma_W, _ = self._low_rank_svd(A, B)
    #             top_value = Sigma_W[0] ** 2
    #             bottom_vector = U_W[:, -1]
    #             top_vector = U_W[:, 0]

    #             # Check that top vector is indeed an eigenvector
    #             WTW = W.T @ W
    #             ratio = WTW @ top_vector / (top_vector * top_value)
    #             torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

    #             # Check that top vector is indeed the top eigenvector
    #             assert (WTW @ top_vector).pow(2).sum() > (WTW @ bottom_vector).pow(
    #                 2
    #             ).sum()

    #             # Save eigenvector and eigvenvalue
    #             vectors_dict[layer][cluster] = top_vector.detach().cpu().numpy()
    #             eigvals_dict[layer][cluster] = top_value.item()

    #     # print(vectors_dict, eigvals_dict)

    #     return vectors_dict, eigvals_dict

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

    # Computing logits
    # logits_mat = torch.zeros(len(experts_prototypes.keys()), current_input.shape[0], current_input.shape[1])
    # for i, expert_name in enumerate(experts_prototypes.keys()):
    #     # current_input shape is: (batch, token_num, 3072)
    #     # experts_prototypes[expert_name] shape is: (3072)
    #     # logits_mat[expert_name] shape is: (batch, token_num)
    #     logits_mat[i] = torch.abs(torch.einsum('btd,d->bt', current_input.to(torch.float32), experts_prototypes[expert_name]))


    # # Assuming `current_input` has shape (batch, token_num, 3072)
    # # Assuming `experts_prototypes` is a dictionary where values have shape (3072,)

    # # Stack the prototypes into a single tensor with shape (num_experts, 3072)
    # prototypes_tensor = torch.stack(list(experts_prototypes.values()))  # Shape: (num_experts, 3072)

    # # Reshape `current_input` for matrix multiplication
    # # New shape: (batch * token_num, 3072)
    # current_input_flat = current_input.view(-1, current_input.shape[-1])

    # # Compute the dot product and take the absolute value
    # # Result shape: (batch * token_num, num_experts)
    # logits_flat = torch.abs(current_input_flat @ prototypes_tensor.T)

    # # Reshape back to (num_experts, batch, token_num)
    # logits_mat = logits_flat.view(current_input.shape[0], current_input.shape[1], -1).permute(2, 0, 1)



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
    # print(experts_matrix.T.shape)
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



