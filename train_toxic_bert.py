import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load and merge your data
comments = pd.read_csv("toxicity_data/test_filtered.csv")
labels = pd.read_csv("toxicity_data/test_labels_cleaned.csv")
df = pd.concat([comments, labels], axis=1)

# Drop rows with -1 labels (optional, if any)
df = df[~(df[label := ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] == -1).any(axis=1)]

small_df = df.sample(n=5000, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    small_df["comment_text"].tolist(), small_df[label].values.tolist(), test_size=0.2, random_state=42
)

# Load tokenizer and tokenize texts
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

# Convert labels to float32
train_labels = np.array(train_labels, dtype=np.float32)
val_labels = np.array(val_labels, dtype=np.float32)

# Create HuggingFace Datasets
train_data = Dataset.from_dict({"text": train_texts, "labels": train_labels.tolist()}).map(tokenize, batched=True)
val_data = Dataset.from_dict({"text": val_texts, "labels": val_labels.tolist()}).map(tokenize, batched=True)

# No need for additional torch.tensor() wrapping now
train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load BERT model with 6 output labels
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6,
    problem_type="multi_label_classification"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert-toxic-output",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
    report = classification_report(labels, preds, target_names=label, output_dict=True, zero_division=0)
    return {f"{key}_f1": report[key]["f1-score"] for key in label}

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate and print report
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
