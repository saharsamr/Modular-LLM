MAX_LENGTH = 4000

# LORA_TARGET_MODU LES = ["o_proj", "qkv_proj"]
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# OTHER_TRAINABLE_MODULES = ['embed_tokens', 'lm_head']
TASK_TYPE = "CAUSAL_LM"

EXPERTS_FOLDER_PATH = 'results/llama'
