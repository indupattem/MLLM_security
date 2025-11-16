#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit Web UI for MLLM Safety Guardrail
Predicts if images are safe for sending to LLMs
"""

import os
import sys
import streamlit as st
from pathlib import Path
import numpy as np

# Setup paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# Page config
st.set_page_config(
    page_title="MLLM Safety Guardrail",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .safe-box {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .unsafe-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 10px 0;
    }
    .neutral-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Load model on startup
@st.cache_resource
def load_model():
    """Load trained toxicity model."""
    try:
        from utils import load_toxicity_model
        model_path = repo_root / 'models' / 'toxicity_classifier'
        if not model_path.exists():
            return None, None, "Model not found. Please train the model first."
        
        model, tokenizer = load_toxicity_model(str(model_path))
        return model, tokenizer, None
    except Exception as e:
        return None, None, f"Error loading model: {e}"

# Main app
def main():
    # Header
    st.title("🛡️ MLLM Safety Guardrail")
    st.markdown("Check if your text/image descriptions are safe for sending to Large Language Models")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Predictions above this threshold are flagged as UNSAFE"
        )
        
        st.divider()
        st.markdown("### About")
        st.info(
            "This tool uses a fine-tuned DistilBERT model trained on the "
            "BeaverTails dataset to predict if text is safe for LLMs.\n\n"
            "Model: DistilBERT-base-uncased\n"
            "Training: 512 examples (80/20 split)\n"
            "Accuracy: 73.79%"
        )
    
    # Load model
    model, tokenizer, error = load_model()
    
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    # Tab interface
    tab1, tab2, tab3 = st.tabs(["Text Input", "Batch Analysis", "Model Info"])
    
    # Tab 1: Single Text Input
    with tab1:
        st.subheader("Check Single Text")
        
        # Input methods
        col1, col2 = st.columns(2)
        
        with col1:
            input_method = st.radio("Input Method", ["Type Text", "Paste Example"])
        
        if input_method == "Type Text":
            user_input = st.text_area(
                "Enter text to check:",
                height=150,
                placeholder="Enter your text here..."
            )
        else:
            examples = {
                "Safe - General Question": "What is machine learning?",
                "Safe - Educational": "How can I improve my coding skills?",
                "Potentially Unsafe - Aggressive": "I want to harm someone.",
                "Potentially Unsafe - Harmful": "How to make explosives?",
            }
            selected_example = st.selectbox("Select an example:", list(examples.keys()))
            user_input = examples[selected_example]
            st.text_area("Example text:", value=user_input, disabled=True, height=100)
        
        if st.button("Check Safety", key="single_check", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text to check.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        from utils import predict_safety
                        result = predict_safety(user_input, model, tokenizer)
                        
                        # Display results
                        st.divider()
                        st.subheader("Results")
                        
                        # Get prediction
                        pred_label = result['label']
                        pred_score = result['score']
                        is_unsafe = pred_score > confidence_threshold
                        
                        # Display with color coding
                        if is_unsafe:
                            st.markdown(f"""
                            <div class="unsafe-box">
                                <h3>🚨 UNSAFE - Not Recommended for LLMs</h3>
                                <p>Confidence: {pred_score:.2%}</p>
                                <p>This content may be harmful or inappropriate.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="safe-box">
                                <h3>✅ SAFE - OK for LLMs</h3>
                                <p>Confidence: {(1-pred_score):.2%}</p>
                                <p>This content appears safe to send to LLMs.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Unsafe Probability", f"{pred_score:.2%}")
                        with col2:
                            st.metric("Safe Probability", f"{(1-pred_score):.2%}")
                        with col3:
                            st.metric("Threshold", f"{confidence_threshold:.2%}")
                        
                        # Show the input
                        with st.expander("View Full Input"):
                            st.write(user_input)
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.subheader("Batch Text Analysis")
        st.write("Upload multiple texts or paste them line-by-line for batch analysis")
        
        batch_input = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="Text 1\nText 2\nText 3\n..."
        )
        
        if st.button("Analyze Batch", type="primary"):
            if not batch_input.strip():
                st.warning("Please enter at least one text.")
            else:
                texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
                
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    try:
                        from utils import batch_predict_safety
                        results = batch_predict_safety(texts, model, tokenizer)
                        
                        st.divider()
                        st.subheader("Batch Results")
                        
                        # Summary stats
                        unsafe_count = sum(1 for r in results if r['score'] > confidence_threshold)
                        safe_count = len(results) - unsafe_count
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Analyzed", len(results))
                        with col2:
                            st.metric("Safe", safe_count, delta=f"{safe_count/len(results)*100:.1f}%")
                        with col3:
                            st.metric("Unsafe", unsafe_count, delta=f"{unsafe_count/len(results)*100:.1f}%")
                        
                        st.divider()
                        
                        # Detailed results table
                        st.subheader("Detailed Results")
                        data = []
                        for i, (text, result) in enumerate(zip(texts, results), 1):
                            status = "🚨 UNSAFE" if result['score'] > confidence_threshold else "✅ SAFE"
                            data.append({
                                "No.": i,
                                "Text": text[:50] + "..." if len(text) > 50 else text,
                                "Status": status,
                                "Unsafe %": f"{result['score']:.1%}",
                                "Safe %": f"{(1-result['score']):.1%}"
                            })
                        
                        st.dataframe(data, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during batch analysis: {e}")
    
    # Tab 3: Model Info
    with tab3:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Model Architecture
            - **Base Model**: DistilBERT (base-uncased)
            - **Task**: Binary Classification (Safe/Unsafe)
            - **Training Data**: BeaverTails dataset
            - **Category**: Animal Abuse responses
            - **Samples Used**: 512 (subsampled)
            - **Train/Test Split**: 80/20 (409 train, 103 test)
            """)
        
        with col2:
            st.markdown("""
            ### Performance Metrics
            - **Accuracy**: 73.79%
            - **Precision**: 58.82%
            - **Recall**: 83.33%
            - **F1 Score**: 0.6897
            - **ROC AUC**: 0.9352
            - **Training Duration**: ~29 minutes (CPU)
            """)
        
        st.divider()
        
        st.markdown("""
        ### Label Definition
        - **SAFE (0)**: Responses appropriate and safe for LLM processing
        - **UNSAFE (1)**: Responses containing harmful, toxic, or inappropriate content
        
        ### Confidence Threshold
        The prediction score represents the probability of being UNSAFE.
        - Score > threshold = UNSAFE (flagged)
        - Score <= threshold = SAFE (allowed)
        
        Adjust the threshold in the sidebar to tune sensitivity.
        """)

if __name__ == "__main__":
    main()
