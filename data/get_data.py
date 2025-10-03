from datasets import load_dataset, concatenate_datasets

# Load your dataset from HF
DATASET_NAME = "emotion"
dataset = load_dataset(DATASET_NAME)  # Replace "emotion" with any dataset name
# Combine all splits into one
all_data = concatenate_datasets([dataset[split] for split in dataset.keys()]) #type: ignore
# Save to CSV
all_data.to_csv(f"{DATASET_NAME}.csv", index=False)
print(f"Saved as {DATASET_NAME}.csv")
