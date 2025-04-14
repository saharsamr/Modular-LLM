MAX_LENGTH = 2000

# LORA_TARGET_MODULES = ["o_proj", "qkv_proj"]
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "dense"] # "q_proj", "k_proj", "v_proj", "dense"
# OTHER_TRAINABLE_MODULES = ['embed_tokens', 'lm_head']
TASK_TYPE = "CAUSAL_LM"

EXPERTS_FOLDER_PATH = 'results'

cluster_checkpoint_names = {}
for i in range(10):
    cluster_checkpoint_names[f"cluster{i}"] = f"/home/tmptildec/Taha/pl2peft/cluster_{i+1}/"

#for i in range(10):
#    cluster_checkpoint_names[f"cluster{i}"] = f"/home/rajabi/phi2_expert_pl2peft_correct_ws/cluster_{i+1}/"

# for i in range(10):
#     cluster_checkpoint_names[f"cluster{i}"] = f"./scripts/results/phi2/cluster{i}_batch1_prop1.0/checkpoint-2000/"
