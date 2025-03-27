# TinyLLM-GPT2
Fine-tuning the DistilGPT2 model on a subset of the WikiText-2 dataset using Hugging Face Transformers. Includes full pipeline: dataset loading, tokenization, training configuration, model training, and text generation.

# 🚀 Fine-Tuning DistilGPT-2 with Hugging Face Transformers

This repository demonstrates how to fine-tune the [DistilGPT-2](https://huggingface.co/distilgpt2) language model using the Hugging Face `transformers` and `datasets` libraries. The training is performed on a small subset (1%) of the [WikiText-2](https://huggingface.co/datasets/wikitext) dataset for quick experimentation and understanding of the fine-tuning pipeline.

---

## 🧠 Project Overview

- Model: `distilgpt2` (lightweight version of GPT-2)
- Dataset: `wikitext-2-raw-v1` (1% of training split)
- Task: Causal Language Modeling (next-token prediction)
- Tokenizer: AutoTokenizer (aligned with GPT2)
- Training framework: Hugging Face `Trainer` API

---

## ⚙️ Training Configuration

| Parameter                  | Value            |
|---------------------------|------------------|
| Epochs                    | 1                |
| Batch Size                | 2 per device     |
| Logging Steps             | 10               |
| Save Steps                | 50               |
| Max Token Length          | 128              |
| Padding Token             | Set to `eos_token` |
| Data Collator             | `DataCollatorForLanguageModeling` (mlm=False) |

---

## 📦 Installation

To get started, install the required libraries:

```bash
pip install torch transformers datasets accelerate
