from merging_lora_modules.base_merging_module import (
    BaseMergingModule,
    cluster_checkpoint_names,
)


class SimpleAveraging(BaseMergingModule):
    def __init__(self, base_model):
        super().__init__(base_model)

    def merge(self):
        self.load_lora_modules()
        self.base_model.add_weighted_adapter(
            adapters=cluster_checkpoint_names.keys(),
            weights=[1/len(cluster_checkpoint_names.keys()) for _ in cluster_checkpoint_names.keys()],
            combination_type='linear',
            adapter_name='simple_average'
        )
        self.base_model.set_adapters('simple_average')
