from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "./model_sentimental"  # path to your trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()  # set model to evaluation mode
# -----------------------------
# CLASS NAMES (CHANGE TO YOUR DATASET LABELS)
# -----------------------------
labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # example for 'emotion' dataset
# -----------------------------
# CLI LOOP
# -----------------------------
print("Type 'exit' to quit.\n")
while True:
    sentence = input("Enter a sentence: ")
    if sentence.lower() in ["exit", "quit"]:
        break
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = labels[probs.argmax().item()] # type: ignore
    print(f"Predicted emotion: {pred_label}\n")
