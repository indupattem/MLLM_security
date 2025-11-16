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
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
try:
    # PEFT imports are optional; used only when QLoRA is enabled
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def load_beavertails_dataset(category="animal_abuse", save_local=True, test_size=0.2):
    """Load the BeaverTails dataset from Hugging Face and split into train/test."""
    print(f"Loading BeaverTails dataset (category: {category})...")
    
    # Load dataset
    dataset = load_dataset("PKU-Alignment/BeaverTails-V", category)
    
    # Check available splits
    print(f"Available splits: {list(dataset.keys())}")
    
    # Get the train split
    if 'train' in dataset:
        full_dataset = dataset['train']
    else:
        # If no 'train' split, use the first available split
        split_name = list(dataset.keys())[0]
        full_dataset = dataset[split_name]
        print(f"Using split: {split_name}")
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Sample keys: {full_dataset[0].keys()}")
    
    # Split into train and test
    print(f"\nCreating train/test split ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
    dataset = full_dataset.train_test_split(test_size=test_size, seed=42)
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    # Optional: save locally for faster reload
    if save_local:
        # Use absolute path to avoid permission issues with relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), "data")
        os.makedirs(data_dir, exist_ok=True)
        dataset_path = os.path.join(data_dir, "beavertails")
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}")
    
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
    output_dir=None,
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    use_qlora: bool = False,
    qlora_r: int = 8,
    qlora_alpha: int = 16,
    qlora_target_modules: list | None = None,
):
    """
    Train a toxicity detection classifier.
    
    Args:
        model_name: Pretrained model to fine-tune
        output_dir: Where to save the trained model (default: ../models/toxicity_classifier)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        use_qlora: Whether to use QLoRA (PEFT + bitsandbytes)
        qlora_r: LoRA rank for QLoRA
        qlora_alpha: LoRA alpha for QLoRA
        qlora_target_modules: Target modules for LoRA (default: ["query", "key", "value", "dense"])
    """
    
    # Resolve output directory with absolute path
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), "models", "toxicity_classifier")
    else:
        output_dir = os.path.abspath(output_dir)
    
    # Load dataset
    dataset = load_beavertails_dataset()
    
    # Load tokenizer
    print(f"\nLoading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Optionally set up QLoRA (4-bit) + PEFT (LoRA) for parameter-efficient fine-tuning
    if use_qlora:
        if get_peft_model is None or prepare_model_for_kbit_training is None:
            raise RuntimeError(
                "QLoRA/PEFT requested but 'peft' (and possibly 'bitsandbytes') are not installed. "
                "Install with: pip install peft bitsandbytes"
            )

        print("Using QLoRA + PEFT (4-bit) for parameter-efficient fine-tuning.")

        # BitsAndBytes quantization config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load model in 4-bit mode
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Prepare model for k-bit training (adds any required wrappers)
        model = prepare_model_for_kbit_training(model)

        # Default target modules for encoder-style models if none provided
        if qlora_target_modules is None:
            qlora_target_modules = ["query", "key", "value", "dense"]

        lora_config = LoraConfig(
            r=qlora_r,
            lora_alpha=qlora_alpha,
            target_modules=qlora_target_modules,
            bias="none",
            task_type="SEQ_CLS",
        )

        model = get_peft_model(model, lora_config)

        # Print number of trainable params for confirmation
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.4f}%)")

    else:
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
        gradient_checkpointing=use_qlora or False,
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
    # Note: output_dir will default to ../models/toxicity_classifier relative to this script
    trainer, results = train_toxicity_classifier(
        model_name="distilbert-base-uncased",
        num_epochs=3,
        batch_size=8,  # Adjust based on your GPU memory
        learning_rate=2e-5
    )
