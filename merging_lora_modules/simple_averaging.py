from merging_lora_modules.base_merging_module import (
    BaseMergingModule,
    cluster_checkpoint_names,
)


class SimpleAveraging(BaseMergingModule):
    def __init__(self, base_model, model_name):
        super().__init__(base_model, model_name)

    def merge(self):
        self.load_lora_modules()
        adapter_names = list(cluster_checkpoint_names.keys())
        self.base_model.add_weighted_adapter(
            adapters=adapter_names,
            weights=[1/len(adapter_names) for _ in adapter_names],
            combination_type='linear',
            adapter_name='simple_average'
        )
        self.base_model.set_adapter('simple_average')
        self.base_model = self.base_model.merge_and_unload()
