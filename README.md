# Comment Toxicity Detection using Deep Learning

This project implements a multi-label classification system to detect various types of toxicity in online comments. The system is built using deep learning architectures (CNN and LSTM) to automate the identification of harmful content and improve online moderation.

## Project Overview
The objective is to classify text data into six categories:
*   **Toxic**
*   **Severe Toxic**
*   **Obscene**
*   **Threat**
*   **Insult**
*   **Identity Hate**

The application provides a real-time interface for text analysis and a comparison dashboard to review model performance metrics.

## Data Preprocessing
The following steps were performed to prepare the dataset for model training:
1.  **Text Cleaning**: Applied regular expressions to remove URLs, special characters, and digits. All text was converted to lowercase.
2.  **Tokenization**: Used the Keras Tokenizer to map words to integer sequences with a vocabulary limit of 20,000 words.
3.  **Padding**: Sequences were padded/truncated to a fixed length of 150 tokens to maintain consistent input dimensions.

## Model Architectures
Two primary deep learning approaches were evaluated:

### 1. Convolutional Neural Network (CNN)
The CNN model uses 1D convolution layers to extract local features and patterns (n-grams) from the text.
*   **Structure**: Embedding -> Conv1D -> GlobalMaxPooling1D -> Dense (ReLU) -> Dropout -> Output (Sigmoid).
*   **Characteristics**: Efficient for keyword detection and pattern matching.

### 2. Long Short-Term Memory (LSTM)
The LSTM model is designed to capture sequential dependencies and long-term context within the sentences.
*   **Structure**: Embedding -> Bidirectional LSTM -> GlobalMaxPooling1D -> Dropout -> Output (Sigmoid).
*   **Characteristics**: Better at understanding the context and flow of natural language compared to standard CNNs.

## Performance Analysis
The models were evaluated using ROC-AUC scores for each label. The LSTM model showed superior performance across most categories by identifying contextual toxicity.

| Category | CNN Accuracy (%) | LSTM Accuracy (%) |
| :--- | :--- | :--- |
| Toxic | 97.24% | 97.67% |
| Severe Toxic | 98.83% | 98.83% |
| Obscene | 98.66% | 98.58% |
| Threat | 94.30% | 95.14% |
| Insult | 98.02% | 98.08% |
| Identity Hate | 95.67% | 96.07% |

**Overall Performance (Mean Accuracy):**
*   CNN: 97.12%
*   LSTM: 97.39% (Selected for production)

## System Implementation
The final application is deployed using **Streamlit**, featuring:
*   **Input Analysis**: Real-time classification with probability scores.
*   **Visualizations**: Bar charts comparing model predictions and category breakdowns.
*   **Model Comparison**: A technical dashboard showing the training results and architecture details.

## Technical Stack
*   **Frameworks**: TensorFlow, Keras 3
*   **Web Framework**: Streamlit
*   **Libraries**: Pandas, NumPy, Matplotlib, Seaborn
*   **Tools**: NLTK, Scikit-learn

## Setup and Usage
1.  Install the required dependencies:
    ```bash
    pip install streamlit tensorflow pandas numpy scikit-learn matplotlib seaborn
    ```
2.  Run the application:
    ```bash
    streamlit run app.py
    ```

---

