# finetune_medqa_fixed.py
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# ------------------------------
# Config
# ------------------------------
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_ID = "bigbio/med_qa"
DATASET_CONFIG = "med_qa_en_source"
OUTPUT_DIR = "./qlora_medqa"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 4
MAX_LENGTH = 512
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

# ------------------------------
# Load dataset
# ------------------------------
ds = load_dataset(DATASET_ID, DATASET_CONFIG, trust_remote_code=True)
train_ds = ds["train"]
val_ds = ds["validation"]

# ------------------------------
# Helper: Format prompt
# ------------------------------
def format_prompt(example):
    question = example["question"]
    options = example.get("options", [])
    opts_txt = "\n".join([f"{opt['key']}) {opt['value']}" for opt in options])
    answer = example["answer"]
    return f"Question: {question}\n{opts_txt}\nAnswer: {answer}"

# ------------------------------
# Preprocess function
# ------------------------------
def preprocess(batch):
    # batch is a dict of lists when batched=True
    inputs = []
    for i in range(len(batch["question"])):
        example = {k: batch[k][i] for k in batch.keys()}
        inputs.append(format_prompt(example))
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# ------------------------------
# Load model and tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Apply LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.train()

# ------------------------------
# Preprocess datasets
# ------------------------------
train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# ------------------------------
# Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    gradient_accumulation_steps=8,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print("✅ Fine-tuning complete. Model saved to", OUTPUT_DIR)
