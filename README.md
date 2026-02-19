# 🚨 Toxicity Guard: AI-Powered Comment Moderation

An advanced Deep Learning application designed to detect and categorize toxic comments across 6 critical labels. This project provides real-time analysis and detailed model performance insights to ensure safer online communities.

---

## 🛡️ Project Overview
In the era of digital communication, toxic behavior in online spaces is a significant challenge. This project leverages **Natural Language Processing (NLP)** and **Sequence Modeling** to automatically detect harmful comments. 

Specifically, it classifies text into six distinct toxicity categories:
- **Toxic**: General rudeness or disrespect.
- **Severe Toxic**: Extreme hostility or hateful language.
- **Obscene**: Use of vulgar or offensive language.
- **Threat**: Intent to inflict harm or violence.
- **Insult**: Personal attacks or derogatory remarks.
- **Identity Hate**: Attacks based on race, religion, gender, or orientation.

---

## 🧬 Data Preprocessing Pipeline
To ensure the model receives high-quality data, the following pipeline was implemented:
1.  **Text Cleaning**: Removal of URLs, special characters, and numbers. Lowercasing and whitespace normalization.
2.  **Tokenization**: Converting words into unique integer tokens using a fitted Keras Tokenizer.
3.  **Vocabulary Management**: Limiting the vocabulary to the top 20,000 most frequent words to reduce noise.
4.  **Sequence Padding**: Standardizing all input sequences to a length of **150 tokens** (`MAX_LEN`) to ensure uniform input for the neural network.

---

## 🧠 Model Architectures
We benchmarked two powerful architectures to find the best balance between speed and contextual understanding:

### 1. CNN (Convolutional Neural Network)
- **Concept**: Uses 1D convolution layers to detect local patterns (n-grams/keywords).
- **Layers**: Embedding → Conv1D → GlobalMaxPooling1D → Dense.
- **Strength**: Extremely fast and great at spotting specific toxic keywords.

### 2. LSTM (Long Short-Term Memory) - **THE WINNER** 🏆
- **Concept**: A type of RNN designed to prevent vanishing gradients and capture long-range dependencies.
- **Layers**: Embedding → LSTM (Bidirectional) → GlobalMaxPooling1D → Dropout → Dense.
- **Strength**: Understands the **flow and context** of a sentence. It can detect toxicity even when no specific "bad words" are used, based on the sentence structure.

---

## � Performance & Evaluation
Multi-label classification (where a comment can be both 'Toxic' and 'Obscene') requires robust metrics. We primarily used **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**.

| Category | CNN Accuracy | LSTM Accuracy |
| :--- | :--- | :--- |
| **Toxic** | 0.965 | **0.972** |
| **Severe Toxic** | 0.988 | **0.991** |
| **Obscene** | 0.975 | **0.981** |
| **Threat** | 0.982 | **0.988** |
| **Insult** | 0.968 | **0.974** |
| **Identity Hate** | 0.973 | **0.979** |

---

## 🖥️ User Interface Features
Built with **Streamlit**, the dashboard provides a premium experience:
- **Real-time Verdicts**: 
    - ✅ **Clean**: Content is safe.
    - ⚠️ **Suspicious**: Moderate risk (20%-50% confidence).
    - 🚩 **Toxic**: High-risk content detected.
- **Interactive Graphs**: Side-by-side comparison of category scores.
- **Model Justification**: A dedicated "Model Insight" tab explaining why LSTM was chosen as the primary engine.

---

## 🛠️ Tech Stack & Dependencies
- **Deep Learning**: TensorFlow 2.15+, Keras 3
- **Web App**: Streamlit
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: Pickle (Tokenizer storage), Regex (Cleaning)

---

## ⚙️ How to Run Locally
1. Clone this project.
2. Install requirements: `pip install streamlit tensorflow pandas numpy scikit-learn matplotlib seaborn`
3. Launch the app: `streamlit run app.py`

---

## 🎓 Learning Outcomes
This project demonstrated the power of **Sequential Models** over traditional keyword-based filters. It successfully handled **Class Imbalance** (toxic comments are much rarer than clean ones) and provided a scalable solution for content moderation.

---
