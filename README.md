# 🛡️ Deep Learning for Comment Toxicity Detection

A comprehensive Deep Learning project that detects toxic comments using **Bidirectional LSTM** neural networks, deployed via a **Streamlit** web application.

---

## 📋 Project Overview

**Domain:** Online Community Management and Content Moderation

**Problem Statement:** Online communities and social media platforms face significant challenges from toxic comments, including harassment, hate speech, and offensive language. This project develops an automated deep learning-based system capable of detecting and classifying toxic comments in real-time.

---

## 🎯 Skills Covered

- ✅ Deep Learning
- ✅ Model Development and Training
- ✅ Model Evaluation and Optimization
- ✅ Natural Language Processing (NLP)
- ✅ Streamlit Web App Development
- ✅ Model Deployment

---

## 🏢 Business Use Cases

| Industry | Application |
|----------|-------------|
| **Social Media Platforms** | Real-time toxic comment filtering |
| **Online Forums** | Automated content moderation |
| **E-learning Platforms** | Safe learning environment creation |
| **News Websites** | Comment section moderation |
| **Brand Safety** | Reputation management and ad placement safety |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras |
| **NLP** | TextVectorization |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **ML Utilities** | Scikit-learn |

---

## 📁 Project Structure

```
Comment Toxicity/
├── 📓 Project_Notebook.ipynb    # Complete training notebook
├── 🌐 app.py                     # Streamlit web application
├── 📄 requirements.txt           # Python dependencies
├── 📊 train.csv                  # Training dataset
├── 📊 test.csv                   # Test dataset
├── 🤖 toxicity_model_end_to_end.h5  # Trained model (generated)
├── 📈 training_history.json      # Training metrics (generated)
├── 📉 label_distribution.png     # Data visualization (generated)
├── 📉 training_curves.png        # Training plots (generated)
├── 📉 roc_curves.png             # ROC curves (generated)
└── 📝 README.md                  # Project documentation
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook Project_Notebook.ipynb
```

Run all cells to:
- Explore and visualize the dataset
- Preprocess text data
- Train the Bidirectional LSTM model
- Evaluate model performance
- Save model artifacts

### Step 3: Launch the Web App

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────┐
│     TextVectorization Layer         │  → Converts text to integer sequences
├─────────────────────────────────────┤
│     Embedding Layer (128 dims)      │  → Dense word representations
├─────────────────────────────────────┤
│     Bidirectional LSTM (64 units)   │  → Captures context both ways
├─────────────────────────────────────┤
│     GlobalMaxPooling1D              │  → Feature extraction
├─────────────────────────────────────┤
│     Dense (256) + Dropout           │  → Classification layers
│     Dense (128) + Dropout           │
├─────────────────────────────────────┤
│     Dense (6, sigmoid)              │  → Multi-label output
└─────────────────────────────────────┘
```

---

## 🏷️ Output Classes

The model predicts 6 types of toxicity:

| Class | Description |
|-------|-------------|
| **Toxic** | General toxic content |
| **Severe Toxic** | Extremely toxic content |
| **Obscene** | Obscene language |
| **Threat** | Threatening content |
| **Insult** | Insulting content |
| **Identity Hate** | Hate based on identity |

---

## 📊 Web App Features

### 🔍 Real-time Analysis
- Enter any comment for instant toxicity prediction
- Visual breakdown of all 6 toxicity categories
- Sample comments for quick testing

### 📂 Batch Analysis
- Upload CSV files for bulk predictions
- Automatic column detection
- Download results as CSV

### 📈 Model Insights
- Training accuracy and loss curves
- Dataset label distribution
- Per-class performance metrics

---

## 📸 Screenshots

*Run the notebook and app to generate visualizations*

---

## 📈 Expected Results

| Metric | Expected Value |
|--------|---------------|
| Validation Accuracy | ~95%+ |
| Validation AUC | ~0.97+ |
| Training Time | ~30-60 min (GPU) |

---

## 🔧 Configuration

Key parameters in the notebook:

```python
MAX_FEATURES = 200000      # Vocabulary size
MAX_SEQUENCE_LENGTH = 1800 # Max comment length
EMBEDDING_DIM = 128        # Embedding dimensions
EPOCHS = 3                 # Training epochs (increase for better results)
BATCH_SIZE = 32            # Batch size
```

---

## 📚 Dataset

The project uses the **Jigsaw/Wikipedia Toxic Comment Classification** dataset containing:
- ~160,000 annotated comments
- Multi-label classification (6 categories)
- Provided as `train.csv` and `test.csv`

---

## 💡 Future Improvements

- [ ] Add BERT/Transformer-based model
- [ ] Implement model explainability (LIME/SHAP)
- [ ] Add real-time API endpoint
- [ ] Multi-language support
- [ ] Fine-tune on domain-specific data

---

## 👤 Author

Built for the GUVI/IIT Madras Deep Learning Internship Project

---

## 📄 License

This project is for educational purposes.
