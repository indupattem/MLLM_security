"""
Train a toxicity detection classifier using the BeaverTails dataset.
This script fine-tunes a transformer model to classify if responses are safe or unsafe.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def load_beavertails_dataset(category="animal_abuse", save_local=True):
    """Load the BeaverTails dataset from Hugging Face."""
    print(f"Loading BeaverTails dataset (category: {category})...")
    
    # Load dataset
    dataset = load_dataset("PKU-Alignment/BeaverTails-V", category)
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    print(f"Sample keys: {dataset['train'][0].keys()}")
    
    # Optional: save locally for faster reload
    if save_local:
        os.makedirs("../data", exist_ok=True)
        dataset.save_to_disk("../data/beavertails")
        print("Dataset saved to ../data/beavertails")
    
    return dataset


def preprocess_data(examples, tokenizer):
    """
    Preprocess dataset: tokenize responses and convert safety labels to binary.
    Label: 0 = safe, 1 = unsafe
    """
    # Tokenize the response text
    tokenized = tokenizer(
        examples['response'], 
        truncation=True, 
        padding=False,  # Dynamic padding handled by data collator
        max_length=512
    )
    
    # Convert safety labels: "yes" -> 0 (safe), "no" -> 1 (unsafe)
    tokenized['labels'] = [0 if label == "yes" else 1 for label in examples['is_response_safe']]
    
    return tokenized


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_toxicity_classifier(
    model_name="distilbert-base-uncased",
    output_dir="../models/toxicity_classifier",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
):
    """
    Train a toxicity detection classifier.
    
    Args:
        model_name: Pretrained model to fine-tune
        output_dir: Where to save the trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
    """
    
    # Load dataset
    dataset = load_beavertails_dataset()
    
    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification: safe vs unsafe
    )
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    results = trainer.evaluate()
    print(f"\nTest Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ Training complete!")
    return trainer, results


if __name__ == "__main__":
    # Train the toxicity classifier
    trainer, results = train_toxicity_classifier(
        model_name="distilbert-base-uncased",
        output_dir="../models/toxicity_classifier",
        num_epochs=3,
        batch_size=8,  # Adjust based on your GPU memory
        learning_rate=2e-5
    )
