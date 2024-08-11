import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load and preprocess the dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations")

# Split the dataset into train and validation sets
dataset = dataset["train"].train_test_split(test_size=0.1)
print(dataset)

def preprocess_function(examples):
    # Combine context and response into input_ids and create dummy labels
    inputs = tokenizer(examples['Context'], examples['Response'], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = [1 if response else 0 for response in examples["Response"]]  # Assuming binary classification for demonstration
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,  # Adjust evaluation steps as necessary
    fp16=True,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model("./medical_conversational_distilbert")
