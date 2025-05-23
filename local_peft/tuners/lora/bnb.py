# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from typing import Any, Optional

import bitsandbytes as bnb
import torch

from local_peft.import_utils import is_bnb_4bit_available, is_bnb_available
from local_peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from local_peft.utils.integrations import dequantize_bnb_weight
from local_peft.utils.other import transpose

from .layer import LoraLayer

from merging_lora_modules.base_merging_module import cluster_checkpoint_names
from merging_lora_modules.arrow_routing import _low_rank_svd, compute_weight

if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            LoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue

                warnings.warn(
                    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                output = dequantize_bnb_weight(weight, state=state)
                if not self.use_dora[active_adapter]:
                    w_data = output.to(lora_data.dtype).to(lora_data.device) + lora_data
                else:
                    # handle dora
                    # since output already includes scaling, set it to 1 here
                    weight_norm = (
                        self.lora_magnitude_vector[active_adapter]
                        .get_weight_norm(output, lora_data, scaling=1)
                        .detach()
                    )
                    # We need to cache weight_norm because it has to be based on the original weights. We
                    # cannot calculate it on the fly based on the merged weights when unmerging because its a
                    # different value
                    self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    w_data = dora_factor.view(-1, 1) * (output + lora_data)

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                output = dequantize_bnb_weight(weight, state=state)

                if not self.use_dora[active_adapter]:
                    w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    w_data = output.data / dora_factor.view(-1, 1) - lora_data

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.weight.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                output = lora_B(lora_A(dropout(sub_batch))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result[sub_batch_indices_list[i]] += output

            return result

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lora_A.weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)

                    if not self.use_dora[active_adapter]:
                        output = lora_B(lora_A(dropout(x))) * scaling
                    else:
                        x = dropout(x)
                        output = self.lora_magnitude_vector[active_adapter](
                            x,
                            lora_A=lora_A,
                            lora_B=lora_B,
                            scaling=scaling,
                            base_layer=self.get_base_layer(),
                        )
                    if requires_conversion:
                        output = output.to(expected_dtype)

                    result = result + output

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep

    def dispatch_bnb_8bit(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)

        return new_module


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            LoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
            )

            self.token_gen_counter = 0
            self.merged_lora_A = None
            self.merged_lora_B = None
            self.experts_prototype = {}

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue

                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)

                output = dequantize_bnb_weight(weight, state=weight.quant_state)
                if not self.use_dora[active_adapter]:
                    w_data = output + lora_data
                else:
                    # handle dora
                    # since output already includes scaling, set it to 1 here
                    weight_norm = (
                        self.lora_magnitude_vector[active_adapter]
                        .get_weight_norm(output, lora_data, scaling=1)
                        .detach()
                    )
                    # We need to cache weight_norm because it has to be based on the original weights. We
                    # cannot calculate it on the fly based on the merged weights when unmerging because its a
                    # different value
                    self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    w_data = dora_factor.view(-1, 1) * (output + lora_data)

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 4-bit linear may get different generations due to rounding errors."
                )

                lora_data = self.get_delta_weight(active_adapter)
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                output = dequantize_bnb_weight(weight, state=weight.quant_state)

                if not self.use_dora[active_adapter]:
                    w_data = output - lora_data
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    w_data = output.data / dora_factor.view(-1, 1) - lora_data

                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            print('Entering _mixed_batch_forward() ...')

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                output = lora_B(lora_A(dropout(sub_batch))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result[sub_batch_indices_list[i]] += output

            return result

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)
            compute_arrow_weights = kwargs.pop("compute_arrow_weights", None)
            top_k = kwargs.pop("top_k", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                if compute_arrow_weights == True:
                    
                    if len(self.experts_prototype) == 0:
                        ## Computing experts prototype

                        for adapter_name in cluster_checkpoint_names.keys():
                            A = self.lora_A[adapter_name].weight.T
                            B = self.lora_B[adapter_name].weight.T
                            
                            # Finding the prototypes in each layer (a.k.a LoRA layer, which is 2 in each model's layer)
                            W = (A @ B).T  # out_features, in_features

                            U_W, Sigma_W, V_W = _low_rank_svd(A, B)
                            top_value = Sigma_W[0] ** 2
                            first_right_singular_vector = V_W.T[:, 0]
                            top_vector = U_W[:, 0]

                            # Check that top vector is indeed an eigenvector
                            WTW = W.T @ W
                            ratio = WTW @ top_vector / (top_vector * top_value)
                            torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

                            # Check that top vector is indeed the top eigenvector
                            # assert (WTW @ top_vector).pow(2).sum() > (WTW @ bottom_vector).pow(
                            #     2
                            # ).sum()
                            
                            self.experts_prototype[adapter_name] = top_vector.detach()


                    arrow_weights = compute_weight(x, self.experts_prototype, top_k).to(device='cuda:0') # shape: (batch, seq_len, expert_num)


                    # weighted_adapter_mat_A = torch.zeros(len(cluster_checkpoint_names.keys()), x.shape[0], 
                    #                                     x.shape[1], self.base_layer.in_features , 8)
                    # weighted_adapter_mat_B = torch.zeros(len(cluster_checkpoint_names.keys()), x.shape[0], 
                    #                                     x.shape[1], 8, self.base_layer.out_features)
                    # for i, adapter_name in enumerate(cluster_checkpoint_names.keys()):
                    #     lora_A = self.lora_A[adapter_name].weight
                    #     lora_B = self.lora_B[adapter_name].weight

                    #     lora_A_per_tok = torch.einsum('bt,dr->btdr', arrow_weights.transpose(1,2)[:,i,:].squeeze(1), lora_A.T)
                    #     lora_B_per_tok = torch.einsum('bt,rd->btrd',arrow_weights.transpose(1,2)[:,i,:].squeeze(1), lora_B.T)
                    #     weighted_adapter_mat_A[i] = lora_A_per_tok
                    #     weighted_adapter_mat_B[i] = lora_B_per_tok



                    # Assuming the following variables are defined:
                    # cluster_checkpoint_names: dict of adapter names
                    # x: input tensor with shape [batch_size, seq_length, ...]
                    # self.base_layer.in_features: input feature size
                    # self.base_layer.out_features: output feature size
                    # arrow_weights: tensor with shape [batch_size, some_dim, num_adapters]
                    # self.lora_A and self.lora_B: dictionaries containing adapter weights

                    # Step 1: Stack all adapter weights
                    adapter_names = list(cluster_checkpoint_names.keys())

                    # Stack lora_A weights: [num_adapters, in_features, 8]
                    lora_A = torch.stack([self.lora_A[name].weight for name in adapter_names], dim=0)

                    # Stack lora_B weights: [num_adapters, 8, out_features]
                    lora_B = torch.stack([self.lora_B[name].weight for name in adapter_names], dim=0)

                    # Step 2: Transpose arrow_weights to [batch_size, num_adapters, seq_length]
                    arrow_weights_transposed = arrow_weights.transpose(1, 2)  # Adjust dimensions as needed

                    # Step 3: Compute weighted_adapter_mat_A and weighted_adapter_mat_B using einsum
                    # weighted_adapter_mat_A: [batch_size, num_adapters, seq_length, in_features, 8]
                    weighted_adapter_mat_A = torch.einsum(
                        'ban,aid->b a n i d',
                        arrow_weights_transposed,      # [batch_size, num_adapters, seq_length]
                        lora_A                         # [num_adapters, in_features, 8]
                    )

                    # weighted_adapter_mat_B: [num_adapters, batch_size, seq_length, 8, out_features]
                    weighted_adapter_mat_B = torch.einsum(
                        'ban,ado->b a n d o',
                        arrow_weights_transposed,      # [batch_size, num_adapters, seq_length]
                        lora_B                         # [num_adapters, 8, out_features]
                    )

                    # Step 4: Permute dimensions to match the desired shape
                    # If needed, adjust the permutation to match [num_adapters, batch_size, seq_length, ..., ...]
                    weighted_adapter_mat_A = weighted_adapter_mat_A.permute(1, 0, 2, 3, 4)  # [num_adapters, batch_size, seq_length, in_features, 8]
                    weighted_adapter_mat_B = weighted_adapter_mat_B.permute(1, 0, 2, 3, 4)  # [num_adapters, batch_size, seq_length, 8, out_features]

                    # Now, weighted_adapter_mat_A and weighted_adapter_mat_B are fully computed without explicit loops


                    # Now we take sum along expert dimension
                    self.merged_lora_A = torch.sum(weighted_adapter_mat_A, dim=0)
                    self.merged_lora_B = torch.sum(weighted_adapter_mat_B, dim=0)
                    
                    

                # Now we should complete the forward path w.r.t the corresponding weights:
                result = self.base_layer(x, *args, **kwargs)
                result = result.clone()

                # Initialising scaling and dropout for the new adapter
                scaling = self.scaling['cluster0']
                dropout = torch.nn.Dropout(p=0.05, inplace=False)
                
                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(self.merged_lora_A.dtype)
                
                x = dropout(x)
                # x shape is: (batch, token_num, model_dim)
                # lora_A shape is: (batch, token_num, model_dim, rank)
                # lora_B shape is: (batch, token_num, rank, model_dim)
                if result.shape[1] == 1:
                    ## Generation case
                    # x = torch.einsum('btd,btdr->btr', x, self.merged_lora_A[:, self.token_gen_counter].unsqueeze(1).to(device='cuda:0'))
                    # x = torch.einsum('btr,btrd->btd', x, self.merged_lora_B[:, self.token_gen_counter].unsqueeze(1).to(device='cuda:0'))
                    # self.token_gen_counter += 1
                    # if self.token_gen_counter >= self.merged_lora_A.shape[1]:
                    #     self.token_gen_counter = 0
                    x = torch.einsum('btd,btdr->btr', x, self.merged_lora_A.mean(1, True).to(device='cuda:0'))
                    x = torch.einsum('btr,btrd->btd', x, self.merged_lora_B.mean(1, True).to(device='cuda:0')) 

                else:
                    # print(x.shape)
                    # print(self.merged_lora_A.shape)
                    # print(self.merged_lora_B.shape)
                    x = torch.einsum('btd,btdr->btr', x, self.merged_lora_A.transpose(2,3).to(device='cuda:0'))
                    x = torch.einsum('btr,btrd->btd', x, self.merged_lora_B.transpose(2,3).to(device='cuda:0')) 

                output = x * scaling

                if requires_conversion:
                    output = output.to(expected_dtype)

                # print('result shape is: {}'.format(result.shape))
                # print('output shape is: {}'.format(output.shape))
                result = result + output

            return result

                # result = self.base_layer(x, *args, **kwargs)
                # # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # # The reason is that in some cases, an error can occur that backprop
                # # does not work on a manipulated view. This issue may be solved with
                # # newer PyTorch versions but this would need extensive testing to be
                # # sure.
                # result = result.clone()

                # for active_adapter in self.active_adapters:
                #     if active_adapter not in self.lora_A.keys():
                #         continue
                #     lora_A = self.lora_A[active_adapter]
                #     lora_B = self.lora_B[active_adapter]
                #     dropout = self.lora_dropout[active_adapter]
                #     scaling = self.scaling[active_adapter]

                #     requires_conversion = not torch.is_autocast_enabled()
                #     if requires_conversion:
                #         expected_dtype = result.dtype
                #         x = x.to(lora_A.weight.dtype)

                #     if not self.use_dora[active_adapter]:
                #         output = lora_B(lora_A(dropout(x))) * scaling
                #     else:
                #         x = dropout(x)
                #         output = self.lora_magnitude_vector[active_adapter](
                #             x,
                #             lora_A=lora_A,
                #             lora_B=lora_B,
                #             scaling=scaling,
                #             base_layer=self.get_base_layer(),
                #         )
                #     if requires_conversion:
                #         output = output.to(expected_dtype)

                #     # print('result shape is: {}'.format(result.shape))
                #     # print('output shape is: {}'.format(output.shape))
                #     result = result + output

            # return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep

    def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
        if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

        return new_module
