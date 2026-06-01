# Customer Feedback Sentiment Analysis using RNN/LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-BiLSTM-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Sequence-based sentiment classification on Amazon customer reviews using a **Bidirectional LSTM** network with trainable word embeddings. Classifies reviews as **positive**, **negative**, or **neutral**.

---

## Architecture

```
Raw Text
   │
   ▼
Text Cleaning (lowercase → HTML strip → stopword removal → lemmatization)
   │
   ▼
Tokenisation + Sequence Padding  (vocab: 10,000 | max_len: 100)
   │
   ▼
Embedding Layer  (64-dim, trainable)
   │
   ▼
Bidirectional LSTM  (64 units, return_sequences=True)
   │
Dropout (0.3)
   │
   ▼
Bidirectional LSTM  (32 units)
   │
Dropout (0.3)
   │
   ▼
Dense (64, ReLU) → Dropout (0.2)
   │
   ▼
Dense (3, Softmax)
   │
   ▼
Sentiment: Positive / Negative / Neutral
```

---

## Dataset

| Attribute     | Detail                                               |
|---------------|------------------------------------------------------|
| Source        | Amazon Product Reviews (Kaggle)                      |
| Task          | 3-class sentiment classification                     |
| Preprocessing | Lowercasing, HTML removal, stopword filtering, lemmatization |
| Tokenisation  | Keras Tokenizer, vocab size 10,000, max length 100   |

---

## Results

| Metric              | Score   |
|---------------------|---------|
| Test Accuracy        | ~87%    |
| Architecture         | BiLSTM  |
| Optimizer            | Adam    |
| Loss                 | Categorical Crossentropy |

Training uses EarlyStopping and ReduceLROnPlateau callbacks to prevent overfitting. Full training curves, per-class precision/recall/F1, and confusion matrix are in the notebook.

---

## Repository Structure

```
Customer-Feedback-Analysis/
│
├── notebooks/
│   └── sentiment_analysis_lstm.ipynb   # Full pipeline: EDA → preprocessing → training → evaluation
│
├── src/
│   └── preprocess.py                   # Text cleaning and tokenisation utilities
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/simonpanigrahi/Customer-Feedback-Analysis.git
cd Customer-Feedback-Analysis
pip install -r requirements.txt
```

---

## Usage

```bash
jupyter notebook notebooks/sentiment_analysis_lstm.ipynb
```

The notebook includes a self-contained demo dataset and runs end-to-end without any external downloads. To use your own Amazon reviews CSV, update the dataset loading cell at the top of the notebook.

**Inference on new text:**
```python
from src.preprocess import clean_text, tokenise_and_pad, load_tokenizer
import numpy as np

tokenizer = load_tokenizer('tokenizer.pkl')
text = "This product is absolutely fantastic!"
seq  = tokenise_and_pad([clean_text(text)], tokenizer, max_len=100)
pred = model.predict(seq)
print(['negative','neutral','positive'][np.argmax(pred)])
```

---

## Tech Stack

Python · TensorFlow/Keras · NLTK · NumPy · Pandas · Matplotlib · Seaborn · Jupyter Notebook

---

## Author

**Simon Kenny Panigrahi**
B.Tech Information Technology, VSSUT Burla (2022–2026)
Former ML Intern, C-DOT — Centre for Development of Telematics, Government of India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-simonkp-blue?logo=linkedin)](https://linkedin.com/in/simonkp)
[![GitHub](https://img.shields.io/badge/GitHub-simonpanigrahi-black?logo=github)](https://github.com/simonpanigrahi)

---

## License

This project is licensed under the MIT License.
