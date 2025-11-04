"""
Real-time guardrail pipeline for content safety filtering.
This module provides a production-ready inference system for toxicity detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_toxicity_model, predict_safety, batch_predict_safety, format_prediction_output


class GuardrailPipeline:
    """
    Production guardrail pipeline for content safety.
    
    Example usage:
        pipeline = GuardrailPipeline()
        result = pipeline.check_content("Is this text safe?")
        if not result['is_safe']:
            print("Content blocked!")
    """
    
    def __init__(self, model_path="../models/toxicity_classifier", threshold=0.7):
        """
        Initialize the guardrail pipeline.
        
        Args:
            model_path: Path to trained model
            threshold: Confidence threshold for blocking content (0.0 - 1.0)
        """
        self.threshold = threshold
        self.model, self.tokenizer = load_toxicity_model(model_path)
        print(f"Guardrail pipeline initialized (threshold: {threshold})")
    
    def check_content(self, text, verbose=False):
        """
        Check if content is safe.
        
        Args:
            text: Input text to check
            verbose: Print detailed output
            
        Returns:
            Dictionary with safety assessment
        """
        result = predict_safety(text, self.model, self.tokenizer, self.threshold)
        
        if verbose:
            print(format_prediction_output(result))
        
        return result
    
    def check_batch(self, texts, verbose=False):
        """
        Check multiple texts at once (more efficient).
        
        Args:
            texts: List of texts to check
            verbose: Print detailed output
            
        Returns:
            List of prediction dictionaries
        """
        results = batch_predict_safety(texts, self.model, self.tokenizer, self.threshold)
        
        if verbose:
            for i, result in enumerate(results, 1):
                print(f"\n--- Text {i} ---")
                print(format_prediction_output(result))
        
        return results
    
    def filter_safe_content(self, texts):
        """
        Filter a list of texts, returning only safe content.
        
        Args:
            texts: List of texts to filter
            
        Returns:
            List of safe texts
        """
        results = self.check_batch(texts)
        safe_texts = [r['text'] for r in results if r['is_safe']]
        return safe_texts
    
    def get_statistics(self, texts):
        """
        Get safety statistics for a collection of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with statistics
        """
        results = self.check_batch(texts)
        
        total = len(results)
        safe_count = sum(1 for r in results if r['is_safe'])
        unsafe_count = total - safe_count
        
        avg_safe_prob = sum(r['safe_probability'] for r in results) / total
        avg_unsafe_prob = sum(r['unsafe_probability'] for r in results) / total
        
        return {
            'total_texts': total,
            'safe_count': safe_count,
            'unsafe_count': unsafe_count,
            'safe_percentage': (safe_count / total) * 100,
            'unsafe_percentage': (unsafe_count / total) * 100,
            'avg_safe_probability': avg_safe_prob,
            'avg_unsafe_probability': avg_unsafe_prob
        }


def demo_pipeline():
    """Demonstrate the guardrail pipeline with examples."""
    
    print("\n" + "="*60)
    print("GUARDRAIL PIPELINE DEMO")
    print("="*60)
    
    # Initialize pipeline
    pipeline = GuardrailPipeline(threshold=0.7)
    
    # Test examples
    test_texts = [
        "Hello! How are you today?",
        "This is a friendly message about helping animals.",
        "I want to discuss animal welfare and protection laws.",
    ]
    
    print("\n🔍 Testing individual texts:")
    print("-" * 60)
    
    for text in test_texts:
        result = pipeline.check_content(text, verbose=False)
        status = "✅ SAFE" if result['is_safe'] else "⚠️ UNSAFE"
        print(f"\n{status} (confidence: {result['confidence']:.2%})")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # Batch processing
    print("\n\n📊 Batch Statistics:")
    print("-" * 60)
    stats = pipeline.get_statistics(test_texts)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Safe: {stats['safe_count']} ({stats['safe_percentage']:.1f}%)")
    print(f"Unsafe: {stats['unsafe_count']} ({stats['unsafe_percentage']:.1f}%)")
    print(f"Average safe probability: {stats['avg_safe_probability']:.2%}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    # Check if model exists
    model_path = "../models/toxicity_classifier"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first by running:")
        print("  python train_toxicity.py")
        sys.exit(1)
    
    # Run demo
    demo_pipeline()
