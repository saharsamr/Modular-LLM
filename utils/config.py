MAX_LENGTH = 4000

LORA_TARGET_MODULES = ["o_proj", "qkv_proj"]
OTHER_TRAINABLE_MODULES = ['embed_tokens', 'lm_head']
TASK_TYPE = "CAUSAL_LM"
