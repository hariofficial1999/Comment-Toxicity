import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
st.set_page_config(page_title="Comment Toxicity Detector", layout="wide", page_icon="🚨")

# --- CUSTOM THEME ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTextArea textarea { border-radius: 10px; }
    .stButton button { width: 100%; border-radius: 20px; background-color: #eb4034; color: white; font-weight: bold; }
    .stButton button:hover { background-color: #c43229; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
# These should match your Data.ipynb settings
MAX_LEN = 150 
TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- FUNCTIONS ---
def clean_text(text):
    # Basic cleaning matching common NLP flows
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_all():
    try:
        # Looking for models saved from Data.ipynb
        model = tf.keras.models.load_model('models/cnn_model.keras')
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except:
        return None, None

# --- UI LAYOUT ---
st.title("🚨 Comment Toxicity Detection")
st.markdown("---")

model, tokenizer = load_all()

if model is None:
    st.error("⚠️ **Model Not Found!**")
    st.info("""
    To make this app work, please add this code to your **Data.ipynb** and run it:
    ```python
    import os, pickle
    os.makedirs('models', exist_ok=True)
    cnn_model.save('models/cnn_model.keras')
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    ```
    """)
else:
    # Sidebar Info
    st.sidebar.title("🛠️ Tools & Info")
    
    # Sample comment selection
    sample_comments = {
        "Custom": "",
        "Clean Comment": "This is a very helpful explanation, thank you for sharing!",
        "Toxic Comment": "You are a complete idiot and your work is absolute garbage.",
        "Obscene Comment": "Shut the f*** up and go away you piece of s***.",
        "Threatening Comment": "I will find you and I will hurt you very badly.",
        "Insulting Comment": "You are a pathetic loser who has nothing better to do.",
        "Identity Hate": "Hating on people because of their race or religion is wrong, but I hate YOU specifically."
    }
    
    selected_sample = st.sidebar.selectbox("📖 Select a Sample Comment:", list(sample_comments.keys()))
    
    st.sidebar.divider()
    st.sidebar.write("This app uses a **CNN** trained on Wikipedia comments from **Data.ipynb**.")
    st.sidebar.success("✅ Model: Ready")
    
    # Input Logic
    default_text = sample_comments[selected_sample]
    user_input = st.text_area("✍️ Enter a comment to analyze:", value=default_text, height=150, placeholder="Type something here...")

    if st.button("Analyze Toxicity"):
        if user_input.strip() == "":
            st.warning("Please type something first!")
        else:
            with st.spinner("Analyzing..."):
                # Preprocess logic from Data.ipynb
                cleaned = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
                
                # Predict
                probs = model.predict(pad)[0]
                
                # Results Display
                st.subheader("📊 Prediction Results")
                
                col1, col2 = st.columns([1, 1.2])
                
                with col1:
                    # Table View
                    results_df = pd.DataFrame({
                        'Labels': TARGET_COLS,
                        'Confidence Score': [f"{p*100:.2f}%" for p in probs]
                    })
                    st.dataframe(results_df, use_container_width=True)

                with col2:
                    # Chart View
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # Use a gradient from green to red based on values
                    color_palette = sns.color_palette("coolwarm", 6)
                    sns.barplot(x=probs*100, y=TARGET_COLS, palette=color_palette, ax=ax)
                    ax.set_xlabel("Confidence (%)")
                    ax.set_title("Toxicity Category Breakdown")
                    st.pyplot(fig)

                # Final Verdict
                max_prob = max(probs)
                if max_prob > 0.5:
                    st.error(f"🚩 **Verdict:** Toxic ({max(probs)*100:.1f}% confidence)")
                else:
                    st.success(f"🛰️ **Verdict:** Clean ({ (1-max_prob)*100:.1f}% safety score)")

st.markdown("---")
st.caption("Developed using logic from **Data.ipynb** for Intern Project.")
