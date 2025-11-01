# src/load_dataset.py

from datasets import load_dataset

# Make sure you are logged in using Hugging Face CLI
# Run in terminal: huggingface-cli login
# Then paste your Hugging Face token

# Load the dataset
ds = load_dataset("PKU-Alignment/BeaverTails-V", "animal_abuse")

# Check dataset splits and sample
print(ds)
print("First sample:", ds['train'][0])

# Optional: save a local copy for faster reload
ds.save_to_disk("../data/bevetrails")
