# GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction

![Overview of GenKnowSub](https://github.com/user-attachments/assets/d0eacdbf-26e0-4921-ad48-eae97adacfa8)

The figure shows the overview of our proposed approach. (a) illustrates the process of training task-specific and general
modules, followed by performing general knowledge subtraction, or GenKnowSub. (b) represents the dynamic task
adaptation stage in a model layer, where the Arrow algorithm selects and combines the most relevant task-specific
modules for each input token.

## 📌 Overview

This repository contains the implementation of **GenKnowSub**, a novel modular framework for large language models (LLMs) that disentangles general knowledge from task-specific adaptations. By subtracting a general knowledge LoRA module from task-specific LoRAs, GenKnowSub isolates task-relevant information, enhancing zero-shot generalization. The framework dynamically combines residual modules using the Arrow routing algorithm for efficient task adaptation.

## 🚀 Key Features

- **General Knowledge Subtraction (GenKnowSub)**: Isolates task-specific knowledge by subtracting a general-domain LoRA from task-specific LoRAs.
- **Dynamic Task Adaptation**: Uses the Arrow routing algorithm to select and combine the most relevant modules for each input token.
- **Multilingual Support**: Evaluated on English, French, and German benchmarks, demonstrating cross-lingual transfer capabilities.
- **Compatibility**: Works with models like Phi-3 and Phi-2, and supports parameter-efficient fine-tuning (PEFT) methods like LoRA.

## 📄 Paper Abstract

Large language models (LLMs) often struggle with zero-shot generalization, and several modular approaches have been proposed to address this challenge. Yet, we hypothesize that a key limitation remains: the entanglement of general knowledge and task-specific adaptations. To overcome this, we propose a modular framework that disentangles these components by constructing a library of task-specific LoRA modules alongside a general-domain LoRA. By subtracting this general knowledge component from each task-specific module, we obtain residual modules that focus more exclusively on task-relevant information a method we call general knowledge subtraction (GenKnowSub). Leveraging the refined task-specific modules and the Arrow routing algorithm (Ostapenko et al., 2024), we dynamically select and combine modules for new inputs without additional training. Our studies on the Phi-3 model and standard Arrow as baselines reveal that using general knowledge LoRAs derived from diverse languages, including English, French, and German, yields consistent performance gains in both monolingual and cross-lingual settings across a wide set of benchmarks. Further experiments on Phi-2 demonstrate how GenKnowSub generalizes to weaker LLMs.

[Read the full paper here](#) 

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saharsamr/Modular-LLM.git
   cd Modular-LLM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Usage

### Training Task-Specific and General Knowledge LoRAs

1. **Train a general knowledge LoRA**:
   ```bash
   # change train_modules.sh like this
   python ../train_experts.py --language_expert=1 --language='en' --le_train_json_path='/content/en_Wiki_10k_LM_511_1.json' --le_test_json_path='/content/en_Wiki_10k_LM_511_1_test.json' --cluster_idx=0 --batch_size=2 --seed=1234
   ```

2. **Train task-specific LoRAs**:
   ```bash
   cd scripts
   chmod 777 train_modules.sh
   ./train_modules.sh
   ```

3. **Perform GenKnowSub** (subtract general LoRA from task LoRAs):
   ```bash
   python merging_modules_vram_control_lang.py --merging_strategy arrow_routing --posthoc_cross_lingual --disentanglement_method subtract --add_functional_only True --dataset_name xnli
   ```

### Dynamic Task Adaptation with Arrow Routing

```bash
python merging_modules_vram_control_lang.py --merging_strategy arrow_routing  --dataset_name xnli
```

## 📊 Results

GenKnowSub improves performance over baseline models and the Arrow routing method across multiple benchmarks:

### English datasets

| Method          | Avg Accuracy (Phi-3) | Avg Accuracy (Phi-2) |
|-----------------|----------------------|----------------------|
| Base Model      | 63.35                | 63.84                |
| Arrow           | 65.56                | 65.15                |
| GenKnowSub (Avg)| **67.17**            | **66.26**            |

### French and German datasets

| Method          | Avg Accuracy (Phi-3) French | Avg Accuracy (Phi-3) German |
|-----------------|----------------------|----------------------|
| Base Model      | 44.08                | 39.64                |
| Arrow           | 44.02                | 42.09                |
| GenKnowSub (Avg)| **47.64**            | **45.99**            |


For detailed results, please take a look at the paper.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please contact:  
- [Your Name](mailto:your.email@example.com)  
- [Paper Authors](#) *(Link to paper for contact details)*  

---
