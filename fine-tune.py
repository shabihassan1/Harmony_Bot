import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_from_disk

# Load the augmented dataset
augmented_dataset = load_from_disk("augmented_dataset")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize the augmented dataset
def preprocess_function(examples):
    return tokenizer(
        examples['input'], truncation=True, padding="max_length", max_length=256
    )

tokenized_dataset = augmented_dataset.map(preprocess_function, batched=True)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust based on dataset size and requirements
    per_device_train_batch_size=16,  # Use appropriate batch size for your hardware
    per_device_eval_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=1000,  # Evaluate every 1000 steps
    fp16=True,  # Enable mixed precision training for faster training
    logging_dir='./logs',
    logging_steps=100,  # Log every 100 steps
    learning_rate=2e-5,  # Fine-tune learning rate if needed
    warmup_steps=500,  # Warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay to prevent overfitting
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # Assuming 'train' split
    eval_dataset=tokenized_dataset["validation"],  # Assuming 'validation' split
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model("./medical_conversational_bert_augmented")