import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# Page Configuration
st.set_page_config(layout="wide", page_title="Comment Toxicity Detection", page_icon="🛡️")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stTextArea textarea { background-color: #262730; color: #ffffff; border-radius: 10px; border: 1px solid #4da6ff; }
    .stButton>button { background: linear-gradient(45deg, #4da6ff, #0066cc); color: white; border-radius: 20px; padding: 10px 24px; font-weight: bold; border: none; }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 15px rgba(77, 166, 255, 0.5); }
    h1 { font-family: 'Helvetica Neue', sans-serif; background: -webkit-linear-gradient(45deg, #00dbde, #fc00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-card { background-color: #1f1f1f; padding: 20px; border-radius: 10px; border-left: 5px solid #4da6ff; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Load Resources
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('toxicity_model_end_to_end.h5')
    except:
        return None

model = load_model()

# Sidebar Navigation
sidebar_option = st.sidebar.radio("Navigation", ["🔍 Real-time Analysis", "📂 Batch Analysis", "📈 Model Insights"])

# --------------------------------------------------------------------------------
# REAL-TIME ANALYSIS TAB
# --------------------------------------------------------------------------------
if sidebar_option == "🔍 Real-time Analysis":
    st.title("🛡️ Real-time Analysis")
    st.markdown("### Detect toxic content instantly")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Comment")
        
        # Sample buttons
        st.markdown("**Try a sample:**")
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        if sample_col1.button("Sample 1 (Safe)"):
            st.session_state.text_input = "I really appreciate your help with this project!"
        if sample_col2.button("Sample 2 (Toxic)"):
            st.session_state.text_input = "You are useless and I hate you."
        if sample_col3.button("Sample 3 (Ambiguous)"):
            st.session_state.text_input = "Why can't you do anything right?"
            
        input_val = st.session_state.get("text_input", "")
        input_text = st.text_area("Enter text to analyze:", value=input_val, height=150)
        
        if st.button("Analyze Toxicity"):
            if model and input_text:
                results = model.predict([input_text])[0]
                cols = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
                
                # Overall Status
                max_score = np.max(results)
                if max_score > 0.5:
                    st.error(f"⚠️ TOXIC CONTENT DETECTED ({max_score:.1%} confidence)")
                else:
                    st.success("✅ CONTENT IS SAFE")
                
                st.markdown("#### Detailed Breakdown")
                for c, score in zip(cols, results):
                    st.write(f"**{c}**")
                    st.progress(float(score))
                    st.caption(f"{score:.1%}")
            elif not model:
                st.error("Model not found. Please train it first.")
            else:
                st.warning("Please enter text.")

    with col2:
        st.info("ℹ️ **About**\nThis tool uses a Deep Learning (Bi-LSTM) model to classify comments into 6 categories of toxicity.")

# --------------------------------------------------------------------------------
# BATCH ANALYSIS TAB
# --------------------------------------------------------------------------------
elif sidebar_option == "📂 Batch Analysis":
    st.title("📂 Batch Analysis")
    st.markdown("### Upload a CSV file for bulk predictions")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and model:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} rows.")
        
        # Column selection
        text_col = st.selectbox("Select the column containing text:", df.columns)
        
        if st.button("Run Predictions"):
            with st.spinner("Processing..."):
                # Batch prediction
                # Ensure input is string
                texts = df[text_col].astype(str).tolist()
                preds = model.predict(texts)
                
                cols = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
                pred_df = pd.DataFrame(preds, columns=cols)
                
                # Combine
                result_df = pd.concat([df, pred_df], axis=1)
                
                st.subheader("Results")
                st.dataframe(result_df.head())
                
                # Download
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "toxicity_predictions.csv", "text/csv")

# --------------------------------------------------------------------------------
# MODEL INSIGHTS TAB
# --------------------------------------------------------------------------------
elif sidebar_option == "📈 Model Insights":
    st.title("📈 Model Insights")
    st.markdown("### Performance Metrics & Data Analysis")
    
    tab1, tab2 = st.tabs(["Training Metrics", "Data Distribution"])
    
    with tab1:
        if os.path.exists('training_history.json'):
            hist = pd.read_json('training_history.json')
            
            st.subheader("Model Accuracy & Loss")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### Accuracy")
                st.line_chart(hist[['accuracy', 'val_accuracy']])
            
            with c2:
                st.markdown("#### Loss")
                st.line_chart(hist[['loss', 'val_loss']])
        else:
            st.warning("No training history found. Please train the model.")
            
    with tab2:
        if os.path.exists('label_distribution.png'):
            st.image('label_distribution.png', caption="Dataset Label Distribution")
        else:
            st.warning("No distribution plot found.")
