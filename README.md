# FundamentalAnalysisGPT - Milestone 3

## Project Overview

FundamentalAnalysisGPT is a custom-built, Pre-LayerNorm Generative Pretrained Transformer (GPT) designed for the financial domain. This project demonstrates a complete end-to-end LLM pipeline consisting of two phases:

1. **Generative Pretraining:** The base model is trained from scratch on the SEC EDGAR corpus (10-K and 10-Q filings) to learn the linguistic structure, vocabulary, and statistical distribution of corporate financial disclosures.
2. **Parameter-Efficient Fine-Tuning (PEFT):** The pre-trained base model is adapted using Low-Rank Adaptation (LoRA) on the FinQA dataset. By freezing 99.86% of the network and injecting trainable rank-8 adapters into the attention projection matrices, the model is taught to shift from "document continuation" to "instruction-following" for financial question answering.

## Hardware Assumptions & Limitations

* **GPU Requirement:** This code is optimized for an **NVIDIA A100** (or equivalent Ampere/Hopper architecture like L4 or H100).
* **Precision & Compilers:** The training loop utilizes `torch.bfloat16` automatic mixed precision, FlashAttention, and `torch.compile`.
* *Note:* Attempting to run this code on older GPUs (like the free Colab T4) will result in a `CUDA error: no kernel image is available for execution` crash due to the lack of hardware support for `bfloat16` tensor cores.

## Environment & Setup Instructions

The project is designed to be run in a Google Colab environment (A100 runtime).

**Critical File Placement:**
Before running the notebook, you **must** upload the `architecture.py` file directly into the root file directory of your Google Colab instance. The Jupyter Notebook relies on this file being in the same root environment to properly import the custom tokenizer, model architecture, and training loops.

**Dependency Installation:**
Colab pre-installs most requirements (like PyTorch and CUDA bindings), but you must strictly downgrade NumPy to avoid C++ ABI conflicts with the HuggingFace datasets library. Run the following command in your environment before executing the code:

```bash
pip install -q "numpy<2" "datasets<3" tiktoken matplotlib

```

*(If running locally, ensure you have PyTorch 2.0+ installed with CUDA 12.1+ support).*

## Directory Structure

All files for this submission are located in the root directory:

* `Milestone_3.ipynb`: The main executable Jupyter Notebook containing the full pipeline.
* `architecture.py`: The consolidated Python module containing the custom Tokenizer, Dataset streaming classes, Transformer model, and optimized training loops.
* `Milestone_3_Report.pdf`: The final report detailing methodology, metrics, and error analysis.
* `loss_curve.png`: The generated plot of the pretraining loss.
* `milestone_2_model.pth` / `lora_adapted_model.pth`: The saved model weights (generated during runtime).

## Exact Commands to Run Training and Evaluation

To reproduce the workflow, open `Milestone_3.ipynb` in Google Colab (with an A100 GPU attached), ensure `architecture.py` is uploaded to the same directory, and execute the cells sequentially.

**To Run Pretraining (Base Model):**

1. In the Configuration cell, set `TRAIN_MODEL = True`.
2. Run the notebook. The script will stream the SEC EDGAR dataset, train for 2,000 steps, and automatically evaluate validation loss every 20 steps.

**To Run LoRA Adaptation:**

1. The notebook automatically freezes the base model and injects the `LinearWithLoRA` wrappers.
2. The FinQA dataset is processed using the `gpt2` tiktoken encoding to maintain vocabulary consistency.
3. The LoRA training loop executes for 500 update steps and prints the training/validation loss.

## Generating Plots, Tables, and Reported Results

All reported metrics and visualizations are generated automatically by the notebook:

* **Loss Curves:** Upon completing the pretraining loop, the notebook uses `matplotlib` to plot the Training vs. Validation loss and saves it directly to the root directory as `loss_curve.png`.
* **Perplexity & Loss Tables:** The final Training Loss, Validation Loss, and calculated Perplexity are printed to the standard output console immediately following the training loops.

## Where Outputs are Saved

* All generated plots and model weights are saved directly to the root directory alongside the notebook.

## How to Reproduce the Demo (Qualitative Evaluation)

The final cells in `Milestone_3.ipynb` contain the inference loops used in the video demo.

To reproduce the base vs. adapted comparison:

1. Ensure the notebook has loaded the `lora_adapted_model.pth` weights (which the script does automatically after fine-tuning).
2. Run the final cell labeled **"Inference Evaluation"**.
3. The script will process 4 distinct financial prompts using multinomial sampling (`temperature=0.8`, `top_k=40`).
4. The generated text will be printed to the console, demonstrating both the model's mastery of SEC structural style and the limitations of its pretraining prior on instruction-following tasks.
