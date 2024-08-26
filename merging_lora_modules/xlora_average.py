from merging_lora_modules.base_merging_module import (
    cluster_checkpoint_names,
)
from transformers import AutoConfig

import xlora
import torch
from data_handler.dataset import read_routing_ds_flan


class XLoraAveraging():
    def __init__(self, base_model, model_name):
        self.base_model = base_model
        self.model_name = model_name
        self.base_model_config = AutoConfig.from_pretrained(
                                    model_name,
                                    trust_remote_code=True,
                                    use_flash_attention_2=False,
                                    device_map="auto",
                                )
        

    def create_model(self):
        self.model = xlora.add_xlora_to_model(
                            model=self.base_model,
                            xlora_config=xlora.xLoRAConfig(
                                self.base_model_config.hidden_size,
                                base_model_id=self.model_name,
                                xlora_depth=4,
                                device=torch.device("cuda"),
                                adapters=cluster_checkpoint_names,
                            ),
                            verbose=True,
                        )


    def train(self):
        ## TODO: Writing the training code for the xlora_model, similar to module_trainer.train() method
        return NotImplementedError

        