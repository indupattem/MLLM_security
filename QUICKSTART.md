# 🚀 MLLM Security - Quick Run Guide

## Fastest Way to Get Started

### Step 1: Activate Environment
```powershell
.\guardrails_env\Scripts\Activate.ps1
```

### Step 2: Install Missing Dependencies (if needed)
```powershell
pip install accelerate huggingface-hub
```

### Step 3: Login to Hugging Face
```powershell
huggingface-cli login
```
Get your token from: https://huggingface.co/settings/tokens

### Step 4: Run Quick Start
```powershell
python src\quickstart.py
```

---

## What Each Option Does

### Option 1: Train Model (Required First!)
- Downloads BeaverTails dataset (~150MB)
- Fine-tunes DistilBERT for toxicity detection
- Takes 10-30 minutes
- Creates `models/toxicity_classifier/`

### Option 2: Evaluate Model
- Tests model on held-out test set
- Generates metrics (accuracy, precision, recall, F1)
- Creates visualization plots in `results/`

### Option 3: Run Demo
- Shows guardrail pipeline in action
- Tests on example texts
- Displays confidence scores

### Option 4: Interactive Testing
- Type your own text to test
- Get instant safety predictions
- Great for experimentation

---

## Manual Commands (Alternative)

### Train
```powershell
python src\train_toxicity.py
```

### Evaluate
```powershell
python src\evaluate_model.py
```

### Demo
```powershell
python src\guardrail_pipeline.py
```

---

## Expected Results

- **Training Accuracy**: 85-95%
- **Test F1 Score**: 0.80-0.90
- **Inference Speed**: ~100-500 texts/second (CPU)

---

## Troubleshooting

### "Model not found"
→ Run Option 1 to train the model first

### "Not logged in to Hugging Face"
→ Run `huggingface-cli login`

### "Import errors"
→ Make sure you activated the environment:
```powershell
.\guardrails_env\Scripts\Activate.ps1
```

### "Out of memory"
→ In `train_toxicity.py`, change `batch_size=8` to `batch_size=4`

---

## Next Steps After Training

1. ✅ Train model (Option 1)
2. ✅ Evaluate performance (Option 2)
3. ✅ Try interactive testing (Option 4)
4. 📝 Use in your own code (see README.md "Use in Your Code" section)
5. 🔧 Adjust threshold in `GuardrailPipeline(threshold=0.7)` for stricter/looser filtering

---

## Project Structure

```
MLLM_security/
├── src/
│   ├── quickstart.py          ⭐ START HERE
│   ├── train_toxicity.py      # Training
│   ├── evaluate_model.py      # Evaluation
│   ├── guardrail_pipeline.py  # Inference
│   └── utils.py               # Helpers
├── models/                    # Saved models (created after training)
├── data/                      # Cached datasets
├── results/                   # Plots and outputs
└── README.md                  # Full documentation
```

---

**Need help?** Check the full README.md for detailed documentation and code examples!
