# MLLM Safety Guardrail - Web UI Guide

## Quick Start

### 1. Start the Web Application

```powershell
python run_app.py
```

Or directly with Streamlit:

```powershell
streamlit run app.py
```

The app will open at: `http://localhost:8501`

### 2. Features

#### Single Text Analysis
- **Type or Paste Text**: Enter any text you want to check
- **Example Selector**: Choose from pre-loaded examples (safe and unsafe)
- **Real-time Prediction**: Get instant safety assessment
- **Confidence Display**: See probability scores for both safe and unsafe predictions
- **Color Coding**: 
  - 🟢 Green = SAFE (OK for LLMs)
  - 🔴 Red = UNSAFE (Not recommended)

#### Batch Analysis
- **Multiple Texts**: Check multiple texts at once (one per line)
- **Statistics**: Get summary of safe/unsafe counts
- **Detailed Table**: See results for each text with individual scores
- **Export Ready**: Results displayed in a clean table format

#### Model Information
- **Architecture Details**: Learn about the base model (DistilBERT)
- **Performance Metrics**: View accuracy, precision, recall, F1, ROC-AUC
- **Training Info**: Understand what data was used
- **Label Definitions**: Know what SAFE and UNSAFE mean

### 3. Configuration

**Confidence Threshold (Sidebar)**
- Default: 0.5 (50%)
- Adjust to tune sensitivity:
  - Lower threshold → More predictions flagged as UNSAFE (stricter)
  - Higher threshold → Fewer predictions flagged as UNSAFE (lenient)

### 4. Understanding Results

#### Output Fields
- **Unsafe Probability**: Score from 0.0 to 1.0 (higher = more likely unsafe)
- **Safe Probability**: Complement of unsafe (safe + unsafe = 1.0)
- **Threshold**: Your configured sensitivity setting
- **Status**: SAFE (green) or UNSAFE (red) based on threshold

#### Example Results

**SAFE Response:**
```
Text: "How can I improve my coding skills?"
Unsafe Probability: 15%
Safe Probability: 85%
Status: ✅ SAFE
```

**UNSAFE Response:**
```
Text: "Instructions for harmful activity"
Unsafe Probability: 92%
Safe Probability: 8%
Status: 🚨 UNSAFE
```

### 5. Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 73.79% |
| Precision | 58.82% |
| Recall | 83.33% |
| F1 Score | 0.6897 |
| ROC AUC | 0.9352 |

**What This Means:**
- **Accuracy**: The model is correct 73.79% of the time
- **Precision**: When flagging as unsafe, it's correct 58.82% of the time
- **Recall**: It catches 83.33% of actually unsafe content
- **ROC AUC**: 0.9352 means excellent discrimination ability

### 6. Use Cases

#### 1. Content Moderation
```python
# Check user-generated content before sending to LLM
text = "User submitted content..."
# App predicts: SAFE or UNSAFE
```

#### 2. API Protection
```python
# Validate requests in an LLM API
request_text = incoming_data
# App predicts: SAFE/UNSAFE
# If UNSAFE, block the request
```

#### 3. Data Pipeline
```python
# Batch check dataset before training/inference
# Upload multiple texts
# Get summary report
```

#### 4. Safety Testing
```python
# Test your guardrail thresholds
# Adjust sensitivity slider
# See how predictions change
```

### 7. Troubleshooting

**Issue**: "Model not found"
- **Solution**: Run training first: `python src/train_wrapper.py`

**Issue**: "Import error for utils"
- **Solution**: Make sure you're running from the correct directory: `cd /path/to/MLLM_security`

**Issue**: Streamlit not installed
- **Solution**: `pip install streamlit`

**Issue**: Slow predictions
- **Solution**: This is normal on CPU. For faster predictions, consider:
  - Using GPU (NVIDIA with CUDA)
  - Running on a smaller batch
  - Using a simpler model

### 8. Advanced Usage

#### Programmatic Access

```python
import os
import sys
os.chdir(r'c:\...\MLLM_security')
sys.path.insert(0, 'src')

from utils import predict_safety, load_toxicity_model

# Load model
model, tokenizer = load_toxicity_model('models/toxicity_classifier')

# Single prediction
result = predict_safety("Your text here", model, tokenizer)
print(f"Score: {result['score']:.2%}, Label: {result['label']}")

# Batch prediction
texts = ["Text 1", "Text 2", "Text 3"]
from utils import batch_predict_safety
results = batch_predict_safety(texts, model, tokenizer)
for r in results:
    print(f"{r['text']}: {r['label']}")
```

#### Adjusting Threshold

```python
# Get prediction
result = predict_safety(text, model, tokenizer)

# Custom threshold (default 0.5)
threshold = 0.7  # More lenient
is_unsafe = result['score'] > threshold
status = "UNSAFE" if is_unsafe else "SAFE"
```

### 9. Tips & Best Practices

1. **Context Matters**: The model was trained on response safety, so it works best for dialogue/response text
2. **Set Appropriate Threshold**: 
   - 0.3-0.5: Strict (flag more content)
   - 0.5-0.7: Balanced
   - 0.7-0.9: Lenient (flag less content)
3. **Batch Processing**: For large datasets, batch prediction is more efficient
4. **Monitor Results**: Regularly check if the model's predictions align with your safety policies
5. **Update Training**: Consider retraining with new data to improve accuracy

### 10. Architecture Diagram

```
User Input (Text)
       ↓
  Tokenizer (DistilBERT)
       ↓
  Embedding + Encoding
       ↓
  DistilBERT Encoder
       ↓
  Classification Head (Linear Layer)
       ↓
  Softmax → [Safe Prob, Unsafe Prob]
       ↓
  Compare with Threshold
       ↓
  Output: SAFE ✅ or UNSAFE 🚨
```

---

**Enjoy using the MLLM Safety Guardrail! 🛡️**
