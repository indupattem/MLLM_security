# MLLM Security Guardrails

This project implements security guardrails for multimodal large language models (LLMs),
focusing on:

- Toxic content detection  
- Prompt injection defense  
- (Future) Hallucination, Misinformation, and Privacy leak detection  

## Project Structure
MLLM_security/
│
├── data/
│   └── bevetrails/                # (optional local copy or cached dataset)
│
├── models/
│   ├── toxicity_guard/            # trained toxicity detection model will be saved here
│   └── prompt_injection_guard/    # trained prompt injection model will be saved here
│
├── src/                           # all your source code lives here
│   ├── train_toxicity.py
│   ├── train_prompt_injection.py
│   ├── evaluate_model.py
│   ├── guardrail_pipeline.py
│   └── utils.py
│
├── outputs/
│   ├── logs/                      # training logs, results
│   └── checkpoints/               # saved model checkpoints during training
│
├── requirements.txt
└── README.md
