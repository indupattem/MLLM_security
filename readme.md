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

### Load BeaverTails Dataset
```bash
python src/train_toxicity.py
```

This will:
- Download the BeaverTails dataset (animal_abuse category)
- Print dataset structure
- Save a local copy to `data/bevetrails/`

### Expected Output
```
DatasetDict({
    train: Dataset({...})
    test: Dataset({...})
})
First sample: {'prompt': '...', 'category': 'animal_abuse', ...}
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

| Component | Status | Description |
|-----------|--------|-------------|
| Dataset Loading | ✅ Implemented | BeaverTails toxicity dataset |
| Toxicity Training | ⏳ TODO | Fine-tune classifier |
| Prompt Injection | ⏳ TODO | Detect adversarial prompts |
| Guardrail Pipeline | ⏳ TODO | Production inference system |
| Evaluation | ⏳ TODO | Metrics and benchmarking |
| Utilities | ⏳ TODO | Helper functions |

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
