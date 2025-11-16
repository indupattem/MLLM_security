#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher script for the Streamlit web UI
"""
import subprocess
import sys
import os

# Change to repo root
repo_root = r'c:\Users\allam\OneDrive - K L University\Documents\MLLM_security[1]\MLLM_security'
os.chdir(repo_root)

print("="*70)
print("MLLM SAFETY GUARDRAIL - WEB UI")
print("="*70)
print("\nStarting Streamlit app...")
print("Open your browser and go to: http://localhost:8501")
print("\nPress Ctrl+C to stop the server")
print("="*70 + "\n")

# Run Streamlit
try:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
except KeyboardInterrupt:
    print("\n\nServer stopped.")
    sys.exit(0)
