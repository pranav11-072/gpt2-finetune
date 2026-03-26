import os
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME     = "gpt2"
DATASET_NAME   = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
OUTPUT_DIR     = "./gpt2-finetuned"
MAX_LENGTH     = 128
EPOCHS         = 3
BATCH_SIZE     = 4

# ── Load tokenizer & model ─────────────────────────────────────────────────
print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# ── Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

# ── Tokenize ────────────────────────────────────────────────────────────────
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

print("Tokenizing...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# ── Data collator ────────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ── Training arguments ──────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=False,
    report_to="none",
)

# ── Trainer ─────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
