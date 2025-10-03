from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# -----------------------------
# CONFIGURATION - Change these!
# -----------------------------
MODEL_NAME = "bert-base-uncased"   # You can change to 'bert-base-uncased', 'roberta-base', etc.
DATASET_NAME = "emotion"                 # HuggingFace dataset, can be local csv too
NUM_LABELS = 6                           # Number of classes in your classification task
BATCH_SIZE = 16
EPOCHS = 5
LR = 5e-5
OUTPUT_DIR = "./model_sentimental"            # Where your trained model will be saved
# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# -----------------------------
# LOAD DATASET
# -----------------------------
# Hugging Face datasets library supports many datasets directly
dataset = load_dataset(DATASET_NAME)
# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
# Apply tokenizer to dataset
encoded_dataset = dataset.map(tokenize, batched=True)
# Set format for PyTorch
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"]) # type: ignore
# -----------------------------
# LOAD MODEL
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
).to(device)
# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",   # Can be 'steps' too
    save_strategy="epoch",
    learning_rate=LR,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy", # Can change to 'f1', etc.
    logging_dir="./logs",
    logging_steps=50,
    fp16=True
)
# -----------------------------
# METRICS FUNCTION
# -----------------------------
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}
# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer, #type: ignore
    compute_metrics=compute_metrics
)
# -----------------------------
# START TRAINING
# -----------------------------
trainer.train()
# -----------------------------
# SAVE MODEL
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete. Model saved at", OUTPUT_DIR)
