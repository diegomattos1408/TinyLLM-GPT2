# 1. Install Required Packages
# pip install torch transformers datasets accelerate

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline,
)
from datasets import load_dataset
import torch
import random
import numpy as np

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 2. Load model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fix pad_token_id warning
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 3. Load dataset (small and lightweight)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# 4. Tokenize dataset
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./llm-output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    logging_dir='./logs',
)

# 6. Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. Train the model
trainer.train()

# 9. Inference (generate text with the fine-tuned model)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Once upon a time", max_length=50, do_sample=True, truncation=True)
print("\nGenerated text:\n", output[0]["generated_text"])

