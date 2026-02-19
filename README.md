# 🚨 Toxicity Guard: AI-Powered Comment Moderation

An advanced Deep Learning application designed to detect and categorize toxic comments across 6 critical labels. This project provides real-time analysis and detailed model performance insights to ensure safer online communities.

---

## 🛡️ Project Overview
This project leverages **Natural Language Processing (NLP)** and **Sequence Modeling** to classify text into six distinct toxicity categories:
- **Toxic** | **Severe Toxic** | **Obscene** | **Threat** | **Insult** | **Identity Hate**

### 🏆 Winning Architecture: LSTM
While both CNN and LSTM architectures were tested, the **LSTM (Long Short-Term Memory)** model was selected as the production standard due to its superior ability to capture word dependencies and long-term context in sentences.
- **LSTM Mean Accuracy**: 0.9739
- **CNN Mean Accuracy**: 0.9712

---

## 🚀 Key Features
- **Real-time Analysis**: Instant toxicity verdicts with "Clean", "Suspicious", or "Toxic" alerts.
- **Model Insight Dashboard**: Category-wise performance comparison between CNN and LSTM architectures.
- **Visual Evidence**: Probability distribution charts for every analyzed comment.
- **System Health Diagnostics**: Streamlined interface focusing on accuracy and reliability.

---

## 🛠️ Tech Stack
- **Languages**: Python 3.12
- **Deep Learning**: TensorFlow, Keras 3
- **Web App**: Streamlit (Premium UI)
- **Data Science**: Pandas, NumPy
- **NLP**: NLTK, Tokenization, Pad-Sequences
- **Visualization**: Matplotlib, Seaborn

---

## 📁 Project Structure
```text
├── app.py                   # Main Streamlit Web Application
├── Data.ipynb               # Training, EDA, and Model Comparison Notebook
├── best_toxicity_model.h5   # Deployed LSTM Winning Model
├── tokenizer.pkl            # Pre-fitted Text Tokenizer
├── Viva_Preparation.md      # Documentation for Interview Preparation
├── viva_to_pdf.py           # Utility to convert MD to PDF
├── train.csv                # Training Data (Kaggle Dataset)
└── README.md                # Project Documentation
```

---

## ⚙️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install streamlit tensorflow pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Model Insight & Comparison
| Feature | CNN Model | LSTM Model |
| :--- | :--- | :--- |
| **Speed** | ⚡ Extremely Fast | 🕒 Moderate (Sequential) |
| **Context** | Local (Keyword focus) | Global (Sequential dependencies) |
| **Best For** | Pattern detection | Sarcasm & Contextual Harm |
| **Mean Accuracy** | **0.9712** | **0.9739 (WINNER)** |

---

## 🎓 Viva & Interview Prep
The project includes a comprehensive `Viva_Preparation.md` file covering:
- **Deep Learning Fundamentals**: Backpropagation, Activation Functions, Vanishing Gradients.
- **NLP Specialized Concepts**: Stemming/Lemmatization, Word Embeddings, Attention mechanisms.
- **Project-Specific Logic**: Why LSTM won over CNN for toxicity detection.

---


