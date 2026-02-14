# 🚨 Comment Toxicity Detection

An AI-powered web application that detects and categorizes toxic comments across 6 different labels using Deep Learning. Designed to make online interactions safer by identifying harmful content in real-time.

---

## 🌟 Project Overview
This project leverages **Natural Language Processing (NLP)** and **Deep Learning** (CNN & LSTM) to classify Wikipedia comments into six toxicity categories:
- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

## 🚀 Key Features
- **Real-time Interface**: Built with Streamlit for a premium user experience.
- **Deep Learning Models**: Comparison between **CNN** (for pattern matching) and **LSTM** (for sequence/context).
- **Interactive Dashboards**: Visual breakdown of toxicity scores using Seaborn and Matplotlib.
- **Pre-trained Performance**: High AUC scores across all categories.

## 🛠️ Tech Stack
- **Core**: Python 3.12
- **Deep Learning**: TensorFlow, Keras 3
- **Web App**: Streamlit
- **Data Science**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **NLP**: NLTK, Regex

## 📁 Project Structure
```text
├── models/                  # Saved .keras models and tokenizers
├── Data.ipynb               # Main Training & EDA Notebook
├── app.py                   # Streamlit Web Application
├── train.csv                # Training Dataset
├── test.csv                 # Test Dataset
└── README.md                # Project Documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/comment-toxicity.git
   cd comment-toxicity
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model** (Optional):
   Open `Data.ipynb` and run all cells to generate the latest `models/cnn_model.keras` and `models/tokenizer.pkl`.

## 🖥️ Running the Application
Once the models are generated in the `models/` folder, run the Streamlit app:
```bash
streamlit run app.py
```

## 📊 Model Comparison Justification
| Feature | CNN Model | LSTM Model |
| :--- | :--- | :--- |
| **Speed** | Fast | Moderate |
| **Context** | Local (Words) | Global (Sentence) |
| **Best For** | Keywords/Patterns | Sarcasm/Slight Harm |
| **Winner** | **98%+ AUC** | **High Context Accuracy** |

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License
This project is licensed under the MIT License.

---
Developed for **Intern Project: Comment Toxicity Detection**.
