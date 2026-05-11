# Milestone 2: Custom Generative Language Model Pretraining

## How to Run Training and Evaluation
1. **Install dependencies:** `pip install torch datasets matplotlib`
2. **Execute the pipeline:** `python training_code/Milestone_2.py`
   *(This script automatically handles dataset mapping, tokenization, model compilation, the training loop, and final evaluation.)*

## Key Hyperparameters
* **Context Length:** 256 tokens
* **Embedding Dimension:** 256
* **Transformer Layers:** 4
* **Attention Heads:** 8
* **Vocabulary Size:** ~47,319 (Custom Regex Tokenizer)
* **Batch Size:** 256
* **Learning Rate:** 5e-4
* **Optimizer:** AdamW
* **Hardware/Precision:** NVIDIA H100 GPU utilizing `bfloat16` (Automatic Mixed Precision) and TF32 enabled.

## Output Locations
Upon successful execution, the script automatically saves artifacts to the following locations:
* **Loss Curve Plot:** `results/loss_curve.png`
* **Model Weights:** `results/milestone_2_model.pth`
* **Evaluation Metrics:** Final Training and Validation Perplexity are printed directly to the console.
* **Error Analysis (Generations):** The 5 text generation prompts and their outputs are printed directly to the console at the end of the run (also documented in `report.md`).
