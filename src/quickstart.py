"""
Quick Start Script for MLLM Security Project
This script provides a simple interface to run the complete pipeline.
"""

import sys
import os


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def check_environment():
    """Check if required packages are installed."""
    print_banner("CHECKING ENVIRONMENT")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 
        'sklearn', 'pandas', 'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:20s} - installed")
        except ImportError:
            print(f"❌ {package:20s} - MISSING")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Missing packages detected!")
        print("Please install requirements:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed!")
    return True


def check_hf_login():
    """Check if user is logged into Hugging Face."""
    print_banner("CHECKING HUGGING FACE LOGIN")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print("❌ Not logged in to Hugging Face")
        print("\nPlease login using:")
        print("  huggingface-cli login")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        return False


def train_model():
    """Train the toxicity detection model."""
    print_banner("TRAINING TOXICITY CLASSIFIER")
    
    print("\n⚙️  This will:")
    print("  1. Download the BeaverTails dataset (~150MB)")
    print("  2. Fine-tune DistilBERT for toxicity detection")
    print("  3. Save the model to models/toxicity_classifier/")
    print("\n⏱️  Estimated time: 10-30 minutes (depending on hardware)")
    
    response = input("\n▶️  Start training? (yes/no): ").lower()
    if response != 'yes':
        print("Training cancelled.")
        return False
    
    print("\n🚀 Starting training...")
    print("-" * 70)
    # Ask whether to use QLoRA (PEFT + bitsandbytes). Only enable if peft/bitsandbytes are installed
    use_qlora = False
    qlora_choice = input("Enable QLoRA (PEFT + bitsandbytes) for parameter-efficient fine-tuning? (yes/no) [no]: ").lower().strip()
    if qlora_choice == 'yes':
        use_qlora = True

    try:
        from train_toxicity import train_toxicity_classifier
        trainer, results = train_toxicity_classifier(use_qlora=use_qlora)
        print("\n✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_model():
    """Evaluate the trained model."""
    print_banner("EVALUATING MODEL")
    
    model_path = "../models/toxicity_classifier"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first (option 1)")
        return False
    
    try:
        from evaluate_model import evaluate_model as eval_func
        results = eval_func(model_path)
        return True
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_pipeline():
    """Run the guardrail pipeline demo."""
    print_banner("RUNNING GUARDRAIL PIPELINE DEMO")
    
    model_path = "../models/toxicity_classifier"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first (option 1)")
        return False
    
    try:
        from guardrail_pipeline import demo_pipeline as demo_func
        demo_func()
        return True
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_test():
    """Interactive testing interface."""
    print_banner("INTERACTIVE TESTING")
    
    model_path = "../models/toxicity_classifier"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first (option 1)")
        return False
    
    try:
        from guardrail_pipeline import GuardrailPipeline
        
        print("\n🔧 Loading model...")
        pipeline = GuardrailPipeline(threshold=0.7)
        
        print("\n✅ Model loaded! Enter text to check (or 'quit' to exit)")
        print("-" * 70)
        
        while True:
            print("\n📝 Enter text to check:")
            text = input("> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not text:
                print("⚠️  Please enter some text")
                continue
            
            result = pipeline.check_content(text, verbose=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Interactive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main menu."""
    
    print_banner("MLLM SECURITY - QUICK START")
    print("\nWelcome to the MLLM Security Guardrails Project!")
    
    # Check environment
    if not check_environment():
        return
    
    # Check HF login
    if not check_hf_login():
        print("\n⚠️  Warning: You need to login to download the dataset")
        response = input("Continue anyway? (yes/no): ").lower()
        if response != 'yes':
            return
    
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("\n1. Train toxicity classifier (required first step)")
        print("2. Evaluate trained model")
        print("3. Run guardrail pipeline demo")
        print("4. Interactive testing")
        print("5. Check setup again")
        print("6. Exit")
        
        choice = input("\n▶️  Select option (1-6): ").strip()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            evaluate_model()
        elif choice == '3':
            demo_pipeline()
        elif choice == '4':
            interactive_test()
        elif choice == '5':
            check_environment()
            check_hf_login()
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n⚠️  Invalid option. Please select 1-6.")


if __name__ == "__main__":
    # Change to src directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
