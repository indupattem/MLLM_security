"""
Evaluate trained toxicity detection models.
This script provides comprehensive evaluation metrics and visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
from utils import load_toxicity_model, batch_predict_safety


def evaluate_model(model_path="../models/toxicity_classifier", category="animal_abuse"):
    """
    Comprehensive evaluation of the toxicity detection model.
    
    Args:
        model_path: Path to the trained model
        category: BeaverTails dataset category to evaluate on
    """
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    model, tokenizer = load_toxicity_model(model_path)
    
    # Load test dataset
    print(f"\n2. Loading test dataset (category: {category})...")
    dataset = load_dataset("PKU-Alignment/BeaverTails-V", category, split='test')
    print(f"   Test samples: {len(dataset)}")
    
    # Prepare data
    texts = dataset['response']
    true_labels = [0 if label == "yes" else 1 for label in dataset['is_response_safe']]
    
    # Make predictions
    print("\n3. Making predictions...")
    batch_size = 32
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        results = batch_predict_safety(batch_texts, model, tokenizer)
        all_predictions.extend([r['prediction'] for r in results])
        all_probabilities.extend([r['unsafe_probability'] for r in results])
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed {min(i+batch_size, len(texts))}/{len(texts)} samples")
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    print("\n4. Computing metrics...")
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-"*60)
    print(classification_report(
        true_labels, 
        predictions, 
        target_names=['Safe', 'Unsafe'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\n" + "-"*60)
    print("CONFUSION MATRIX")
    print("-"*60)
    print(f"                Predicted Safe    Predicted Unsafe")
    print(f"Actually Safe        {cm[0][0]:6d}           {cm[0][1]:6d}")
    print(f"Actually Unsafe      {cm[1][0]:6d}           {cm[1][1]:6d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n" + "-"*60)
    print("DETAILED METRICS")
    print("-"*60)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f} (of predicted unsafe, how many are truly unsafe)")
    print(f"Recall:      {recall:.4f} (of truly unsafe, how many detected)")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f} (of truly safe, how many detected)")
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(true_labels, probabilities)
        print(f"ROC AUC:     {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC AUC:     Could not compute ({e})")
        roc_auc = None
    
    # Create visualizations
    print("\n5. Generating visualizations...")
    create_evaluation_plots(true_labels, predictions, probabilities, roc_auc)
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def create_evaluation_plots(true_labels, predictions, probabilities, roc_auc=None):
    """Create evaluation visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(true_labels, predictions)
    ax = axes[0, 0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Safe', 'Unsafe'],
           yticklabels=['Safe', 'Unsafe'],
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # 2. ROC Curve
    if roc_auc is not None:
        ax = axes[0, 1]
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax = axes[1, 0]
    precision, recall, _ = precision_recall_curve(true_labels, probabilities)
    ax.plot(recall, precision, color='blue', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # 4. Prediction Distribution
    ax = axes[1, 1]
    safe_probs = probabilities[true_labels == 0]
    unsafe_probs = probabilities[true_labels == 1]
    
    ax.hist(safe_probs, bins=50, alpha=0.5, label='Actually Safe', color='green')
    ax.hist(unsafe_probs, bins=50, alpha=0.5, label='Actually Unsafe', color='red')
    ax.set_xlabel('Unsafe Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("../results", exist_ok=True)
    plot_path = "../results/evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Plots saved to {plot_path}")
    
    # Try to show plot (may not work in all environments)
    try:
        plt.show()
    except:
        print("   (Display not available - plot saved to file)")


if __name__ == "__main__":
    # Check if model exists
    model_path = "../models/toxicity_classifier"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first by running:")
        print("  python train_toxicity.py")
        sys.exit(1)
    
    # Run evaluation
    results = evaluate_model(model_path)
