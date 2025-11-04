# MLLM Security - Multi-Modal Language Model Safety Guardrails

## Overview
This project focuses on building and evaluating **safety guardrails** for Multi-Modal Language Models (MLLMs). The goal is to train classifiers that can detect and prevent harmful content, including:

- **Toxicity Detection**: Identifying toxic, harmful, or offensive language
- **Prompt Injection Detection**: Detecting adversarial prompts that attempt to manipulate model behavior
- **Content Safety**: Using datasets like BeaverTails for safety alignment

## Project Structure
```
MLLM_security/
├── src/
│   ├── train_toxicity.py           # Toxicity detection dataset loader (BeaverTails)
│   ├── train_prompt_injection.py   # Prompt injection attack detection (TODO)
│   ├── guardrail_pipeline.py       # End-to-end guardrail inference pipeline (TODO)
│   ├── evaluate_model.py           # Model evaluation metrics (TODO)
│   └── utils.py                    # Helper functions (TODO)
├── data/                            # Local dataset storage
├── guardrails_env/                  # Python virtual environment (not tracked in git)
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

## What the Project Does

### Current Implementation
1. **Dataset Loading** (`train_toxicity.py`):
   - Loads the **BeaverTails** dataset from PKU-Alignment
   - Focuses on animal abuse category as an example
   - Saves dataset locally for faster access
   - Requires Hugging Face authentication

### Planned Components (TODO)
2. **Training Scripts**:
   - Fine-tune transformer models for toxicity classification
   - Train prompt injection detectors
   
3. **Guardrail Pipeline**:
   - Real-time content filtering
   - Multi-stage safety checks
   - Confidence scoring and thresholding

4. **Evaluation**:
   - Precision, recall, F1 metrics
   - False positive/negative analysis
   - Cross-dataset validation

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/indupattem/MLLM_security.git
cd MLLM_security
```

### 2. Create a Virtual Environment
```powershell
# PowerShell (Windows)
python -m venv guardrails_env
.\guardrails_env\Scripts\Activate.ps1

# Or use the existing environment if already created
.\guardrails_env\Scripts\Activate.ps1
```

```bash
# Linux/Mac
python -m venv guardrails_env
source guardrails_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Authenticate with Hugging Face
The BeaverTails dataset requires authentication:
```bash
huggingface-cli login
# Enter your Hugging Face token when prompted
```

You can get a token from: https://huggingface.co/settings/tokens

## Usage

### Quick Start (Recommended)
The easiest way to get started is using the interactive quick start script:

```powershell
# Activate your environment
.\guardrails_env\Scripts\Activate.ps1

# Run the quick start menu
python src\quickstart.py
```

This will guide you through:
1. ✅ Environment verification
2. ✅ Training the model
3. ✅ Evaluation
4. ✅ Interactive testing

### Manual Usage

#### 1. Train the Toxicity Classifier
```powershell
# This will take 10-30 minutes depending on your hardware
python src\train_toxicity.py
```

**What it does:**
- Downloads BeaverTails dataset (~150MB)
- Fine-tunes DistilBERT for binary classification (safe vs unsafe)
- Saves model to `models/toxicity_classifier/`
- Reports training metrics (accuracy, F1, etc.)

**Output:** Trained model with ~85-95% accuracy

#### 2. Evaluate the Model
```powershell
python src\evaluate_model.py
```

**What it does:**
- Loads the trained model
- Evaluates on test set
- Generates classification report
- Creates visualization plots (saved to `results/`)

**Output:** Metrics, confusion matrix, ROC curve, PR curve

#### 3. Run Guardrail Pipeline Demo
```powershell
python src\guardrail_pipeline.py
```

**What it does:**
- Loads trained model
- Tests on example texts
- Shows real-time safety predictions
- Provides batch statistics

**Output:** Safety predictions with confidence scores

#### 4. Use in Your Code
```python
from guardrail_pipeline import GuardrailPipeline

# Initialize pipeline
pipeline = GuardrailPipeline(threshold=0.7)

# Check single text
result = pipeline.check_content("Your text here")
if not result['is_safe']:
    print(f"⚠️ Unsafe content detected! (confidence: {result['confidence']:.2%})")

# Check multiple texts (more efficient)
texts = ["Text 1", "Text 2", "Text 3"]
results = pipeline.check_batch(texts)

# Filter safe content only
safe_texts = pipeline.filter_safe_content(texts)

# Get statistics
stats = pipeline.get_statistics(texts)
print(f"Safe: {stats['safe_percentage']:.1f}%")
```

## Dependencies
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformer models
- **datasets**: Dataset loading and processing
- **evaluate**: Model evaluation metrics
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **tqdm**: Progress bars
- **jupyter**: Interactive notebooks
- **matplotlib**: Data visualization

## Development Status

| Component | Status | File | Description |
|-----------|--------|------|-------------|
| Dataset Loading | ✅ Complete | `train_toxicity.py` | BeaverTails dataset loader |
| Model Training | ✅ Complete | `train_toxicity.py` | Fine-tune DistilBERT classifier |
| Guardrail Pipeline | ✅ Complete | `guardrail_pipeline.py` | Production inference system |
| Evaluation | ✅ Complete | `evaluate_model.py` | Metrics and visualizations |
| Utilities | ✅ Complete | `utils.py` | Helper functions |
| Quick Start | ✅ Complete | `quickstart.py` | Interactive menu system |
| Prompt Injection | ⏳ TODO | `train_prompt_injection.py` | Adversarial prompt detection |

## Project Files Overview

```
src/
├── quickstart.py              # 🚀 START HERE - Interactive menu
├── train_toxicity.py          # Training script for toxicity classifier
├── guardrail_pipeline.py      # Real-time content filtering pipeline
├── evaluate_model.py          # Model evaluation and metrics
├── utils.py                   # Helper functions (model loading, prediction)
└── train_prompt_injection.py # TODO: Prompt injection detection

models/
└── toxicity_classifier/       # Saved model (created after training)
    ├── config.json
    ├── model.safetensors
    └── tokenizer files

data/
└── beavertails/              # Cached dataset (created after first run)

results/
└── evaluation_plots.png      # Visualization outputs
```

## Issues Resolved
- ✅ Added `.gitignore` to prevent tracking virtual environment
- ✅ Virtual environment (`guardrails_env/`) is now ignored by git
- ✅ Project structure documented

## Next Steps
1. **Implement Training Script**:
   - Add fine-tuning logic for toxicity classification
   - Use DistilBERT or RoBERTa as base model
   - Save trained model checkpoints

2. **Build Guardrail Pipeline**:
   - Load trained models
   - Implement real-time inference
   - Add confidence thresholds

3. **Add Evaluation**:
   - Create test harness
   - Compute metrics on held-out test set
   - Generate confusion matrices

4. **Documentation**:
   - Add training tutorials
   - Create Jupyter notebooks with examples
   - Document model performance

## References
- [BeaverTails Dataset](https://huggingface.co/datasets/PKU-Alignment/BeaverTails-V)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PKU-Alignment Research](https://github.com/PKU-Alignment)

## License
(Add your license here)

## Contributors
(Add contributors here) 
