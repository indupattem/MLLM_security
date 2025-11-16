"""
Check training progress and model save status.
This script monitors if the training is running and reports the current state.
"""

import os
import json
import time
from pathlib import Path


def get_repo_root():
    """Get the root directory of the repository."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def check_model_files():
    """Check if model files have been saved."""
    repo_root = get_repo_root()
    model_dir = os.path.join(repo_root, "models", "toxicity_classifier")
    
    if not os.path.exists(model_dir):
        return {
            "status": "NOT_STARTED",
            "message": f"Model directory not found: {model_dir}",
            "files": []
        }
    
    # List all files in the model directory
    files = []
    total_size = 0
    
    try:
        for root, dirs, filenames in os.walk(model_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                size = os.path.getsize(filepath)
                total_size += size
                rel_path = os.path.relpath(filepath, model_dir)
                files.append({
                    "name": rel_path,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "modified": time.ctime(os.path.getmtime(filepath))
                })
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Error reading model directory: {e}",
            "files": []
        }
    
    if not files:
        return {
            "status": "TRAINING_IN_PROGRESS",
            "message": "Model directory exists but is empty (training may be in progress)",
            "files": files,
            "total_size_mb": 0
        }
    
    return {
        "status": "MODEL_SAVED",
        "message": "Model files found - training completed!",
        "files": files,
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }


def check_dataset_cache():
    """Check if dataset has been cached."""
    repo_root = get_repo_root()
    dataset_dir = os.path.join(repo_root, "data", "beavertails")
    
    if not os.path.exists(dataset_dir):
        return {"status": "NOT_CACHED", "message": "Dataset cache not found"}
    
    # Count files and get size
    total_size = 0
    file_count = 0
    
    try:
        for root, dirs, filenames in os.walk(dataset_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                total_size += os.path.getsize(filepath)
                file_count += 1
    except Exception as e:
        return {"status": "ERROR", "message": f"Error reading dataset: {e}"}
    
    return {
        "status": "CACHED",
        "message": f"Dataset cache found with {file_count} files",
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "file_count": file_count
    }


def check_training_logs():
    """Check if training logs exist."""
    repo_root = get_repo_root()
    log_dir = os.path.join(repo_root, "models", "toxicity_classifier", "logs")
    
    if not os.path.exists(log_dir):
        return {"status": "NO_LOGS", "message": "No training logs directory found"}
    
    log_files = []
    try:
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath):
                log_files.append({
                    "name": filename,
                    "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2)
                })
    except Exception as e:
        return {"status": "ERROR", "message": f"Error reading logs: {e}"}
    
    return {
        "status": "LOGS_FOUND",
        "message": f"Found {len(log_files)} log files",
        "log_files": log_files
    }


def main():
    """Main progress check."""
    print("\n" + "="*70)
    print("TRAINING PROGRESS CHECK")
    print("="*70)
    
    print("\n📊 Model Status:")
    print("-" * 70)
    model_status = check_model_files()
    print(f"  Status: {model_status['status']}")
    print(f"  Message: {model_status['message']}")
    if model_status['files']:
        print(f"  Total Size: {model_status['total_size_mb']} MB")
        print(f"  Files:")
        for file_info in model_status['files']:
            print(f"    - {file_info['name']}: {file_info['size_mb']} MB (modified: {file_info['modified']})")
    
    print("\n📦 Dataset Cache Status:")
    print("-" * 70)
    dataset_status = check_dataset_cache()
    print(f"  Status: {dataset_status['status']}")
    print(f"  Message: {dataset_status['message']}")
    if 'total_size_mb' in dataset_status:
        print(f"  Size: {dataset_status['total_size_mb']} MB")
        print(f"  Files: {dataset_status['file_count']}")
    
    print("\n📝 Training Logs:")
    print("-" * 70)
    log_status = check_training_logs()
    print(f"  Status: {log_status['status']}")
    print(f"  Message: {log_status['message']}")
    if 'log_files' in log_status:
        for log_file in log_status['log_files']:
            print(f"    - {log_file['name']}: {log_file['size_mb']} MB")
    
    print("\n" + "="*70)
    
    # Summary and recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 70)
    
    if model_status['status'] == "MODEL_SAVED":
        print("✅ Training completed! Model is ready.")
        print("   Next steps:")
        print("   1. Run evaluation: python src/evaluate_model.py")
        print("   2. Test guardrail pipeline: python src/quickstart.py (option 3)")
        print("   3. Run interactive test: python src/quickstart.py (option 4)")
    
    elif model_status['status'] == "TRAINING_IN_PROGRESS":
        print("⏳ Training is in progress...")
        print("   The model directory exists but is still being written to.")
        print("   Check back again in a few minutes.")
        print("   Estimated total time: 10-30 minutes on CPU, 2-5 minutes on GPU")
    
    elif model_status['status'] == "NOT_STARTED":
        print("❌ Training has not started or was not successful.")
        print("   1. Check if the training process is running")
        print("   2. Run training again: python src/train_toxicity.py")
        print("   3. Or use quickstart: python src/quickstart.py (option 1)")
    
    if dataset_status['status'] == "CACHED":
        print(f"\n✅ Dataset is cached ({dataset_status['total_size_mb']} MB)")
        print("   This means re-running training will be faster!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
