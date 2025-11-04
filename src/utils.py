"""
Utility functions for MLLM security guardrails.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_toxicity_model(model_path="../models/toxicity_classifier"):
    """
    Load a trained toxicity detection model.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    
    return model, tokenizer


def predict_safety(text, model, tokenizer, threshold=0.5):
    """
    Predict if a text is safe or unsafe.
    
    Args:
        text: Input text to classify
        model: Trained classification model
        tokenizer: Tokenizer for the model
        threshold: Confidence threshold for unsafe classification
        
    Returns:
        dict with prediction, confidence, and label
    """
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    # Get prediction and confidence
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()
    
    # Label mapping: 0 = safe, 1 = unsafe
    is_safe = predicted_class == 0
    label = "SAFE" if is_safe else "UNSAFE"
    
    return {
        "text": text,
        "prediction": predicted_class,
        "label": label,
        "confidence": confidence,
        "is_safe": is_safe,
        "safe_probability": probs[0][0].item(),
        "unsafe_probability": probs[0][1].item()
    }


def batch_predict_safety(texts, model, tokenizer, threshold=0.5):
    """
    Predict safety for a batch of texts.
    
    Args:
        texts: List of input texts
        model: Trained classification model
        tokenizer: Tokenizer for the model
        threshold: Confidence threshold for unsafe classification
        
    Returns:
        List of prediction dictionaries
    """
    device = next(model.parameters()).device
    
    # Tokenize inputs
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    # Process results
    results = []
    for i, text in enumerate(texts):
        predicted_class = torch.argmax(probs[i]).item()
        confidence = probs[i][predicted_class].item()
        is_safe = predicted_class == 0
        label = "SAFE" if is_safe else "UNSAFE"
        
        results.append({
            "text": text,
            "prediction": predicted_class,
            "label": label,
            "confidence": confidence,
            "is_safe": is_safe,
            "safe_probability": probs[i][0].item(),
            "unsafe_probability": probs[i][1].item()
        })
    
    return results


def format_prediction_output(prediction_result):
    """
    Format prediction result for display.
    
    Args:
        prediction_result: Dictionary from predict_safety
        
    Returns:
        Formatted string
    """
    result = prediction_result
    output = f"""
{'='*60}
TOXICITY DETECTION RESULT
{'='*60}
Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}

Prediction: {result['label']}
Confidence: {result['confidence']:.2%}

Probabilities:
  - Safe:   {result['safe_probability']:.2%}
  - Unsafe: {result['unsafe_probability']:.2%}
{'='*60}
"""
    return output
