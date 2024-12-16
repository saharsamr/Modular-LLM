MAX_LENGTH = 4000

# LORA_TARGET_MODULES = ["o_proj", "qkv_proj"]
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "dense"]
# OTHER_TRAINABLE_MODULES = ['embed_tokens', 'lm_head']
TASK_TYPE = "CAUSAL_LM"

EXPERTS_FOLDER_PATH = 'results'
