

# SFTA: LLM-Based Bug Triaging

This repository implements SFTA (Supervised Fine-Tuned Assigner), a system that fine-tunes a large language model for automated issue assignmnt and developer recommendation.
It includes dataset preparation scripts, a Jupyter notebook for fine-tuning and evaluation, and configuration files for reproducible experiments.

---


## ⚙️ Installation

```bash
git clone https://github.com/<username>/SFTA.git
cd SFTA
pip install -r requirements.txt
```

**Requirements:**

* Python 3.10+
* CUDA-enabled GPU (recommended)
* PyTorch ≥ 2.0.0

---

# Data Preparation

datasets could be downloaded in: https://github.com/ArshamKhosravani/test_BugTriage/tree/master/dl4ba-main/datasets

# Model Fine-Tuning

The **bug_triage_notebook.ipynb** file contains the full training pipeline using **DeepSeek-R1-Distill-Llama-8B** as the base model.

It demonstrates:

* Loading the processed JSONL data with `datasets`.
* Fine-tuning the model using **causal language modeling (CLM)** loss.
* Applying configurations for gradient checkpointing, AdamW optimizer, and bfloat16 precision.
* Evaluating with **Hit@K (K = 1–10)** to measure developer ranking performance.

Run in Jupyter Lab:

```bash
jupyter lab bug_triage_notebook.ipynb
```

---

## Dependencies

Main dependencies listed in `requirements.txt`:

```text
transformers>=4.48.3
trl>=0.5.1
datasets>=2.13.1
torch>=2.0.0
unsloth==2025.2.12
pandas>=2.1.0
tqdm>=4.66.1


Would you like me to append a short **“Training Configuration Example”** block (with sample hyperparameters like learning rate, batch size, epochs) at the end of the README?
