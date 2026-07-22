# 🧠 Next Word Predictor — Bidirectional LSTM

> **NLP project** that predicts the next word in a sentence using a **Bidirectional LSTM** neural network, deployed as a live **Streamlit** web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red?logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-BiLSTM-green)
![Tests](https://img.shields.io/badge/Tests-66%20passed-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

This project implements a complete **end-to-end NLP pipeline** — from raw text to a live prediction web app:

1. **Data Pipeline** — Text cleaning, OOV tokenization, prefix n-gram sequences
2. **Model Training** — BiLSTM with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
3. **Evaluation** — Train/Val/Test accuracy, Top-3 accuracy, Perplexity
4. **Live App** — Streamlit web app with real-time predictions and confidence scores

### Why this project?

Next-word prediction powers:
- 📱 Mobile keyboard autocomplete (Gboard, SwiftKey)
- 🤖 ChatGPT and large language models
- 📧 Gmail Smart Compose
- 🔍 Google Search suggestions

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Live Prediction** | Real-time next-word suggestions as you type |
| 📊 **Confidence Bars** | Visual probability for each prediction |
| 👆 **Click to Append** | Click any prediction to add it to input |
| 🤖 **Auto-Generate** | Generate full sentences automatically |
| 💬 **Chat & Train** | Build corpus through chat, retrain model |
| 💾 **Auto Save/Load** | Model persists between sessions |
| 🧪 **66 Unit Tests** | Full test coverage across all modules |
| 🔧 **11 Bugs Fixed** | Fixed all critical bugs from original code |

---

## 🏗 Model Architecture

```
Input: token sequence (pre-padded)
         │
         ▼
Embedding(vocab_size × 128)
         │
         ▼
Bidirectional(LSTM(256, return_sequences=True))
         │
    Dropout(0.3)
         │
         ▼
LSTM(128)
         │
    Dropout(0.3)
         │
         ▼
Dense(vocab_size, softmax)
         │
         ▼
Top-K predictions with confidence %
```

| Hyperparameter | Value | Why |
|---|---|---|
| Embedding dim | 128 | Best for ≤10K vocab |
| BiLSTM units | 256 → 128 | Rich bidirectional context |
| Dropout | 0.3 | Prevents overfitting |
| Loss | `categorical_crossentropy` | y is one-hot encoded |
| Optimizer | Adam (lr=0.001) | LR reduces on plateau |
| EarlyStopping | patience=12 | Stops before overfitting |
| Batch size | 32 | Laptop-friendly |

---

## 🐛 11 Critical Bugs Fixed

| # | Original Bug | Fix Applied |
|---|---|---|
| 1 | `GRU` not imported — `NameError` | Replaced with `LSTM` |
| 2 | Fixed 4-token window — padding useless | Proper prefix n-gram |
| 3 | `sparse_categorical_crossentropy` with one-hot y | `categorical_crossentropy` |
| 4 | Dead code after `return` — `NameError` | Removed dead block |
| 5 | `char_history` typo — `NameError` | Fixed to `chat_history` |
| 6 | `texts_to_squences` typo — `AttributeError` | Fixed spelling |
| 7 | `range(token_list)` — `TypeError` | Fixed to `range(len(...))` |
| 8 | `y = sequences[:,:-1]` — trains on wrong labels | `y = padded[:,-1]` |
| 9 | O(n²) vocab rebuild per prediction | O(1) `index_word` lookup |
| 10 | Training fires on every message — race condition | Threshold + thread lock |
| 11 | `index_word` not saved to disk | Both dicts persisted |

---

## 📁 Project Structure

```
Next_Word_Predictor_Streamlit/
│
├── streamlit_app.py          ← Entry point
├── config.py                 ← All constants (single source of truth)
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   ├── data/
│   │   └── preprocessor.py  ← Clean, tokenize, generate sequences
│   ├── model/
│   │   ├── architecture.py  ← BiLSTM model
│   │   ├── trainer.py       ← Training + evaluation pipeline
│   │   ├── predictor.py     ← Prediction with confidence %
│   │   └── persistence.py   ← Save/load model + tokenizer
│   └── utils/
│       └── helpers.py       ← Perplexity, plots, formatting
│
├── data/
│   └── download_corpus.py   ← Downloads from Project Gutenberg
│
├── models/                   ← Saved weights (gitignored)
├── results/                  ← Training plots (gitignored)
└── tests/
    └── test_pipeline.py     ← 66 pytest tests
```

---

## ⚙️ Installation

```bash
# 1. Clone
git clone https://github.com/Rajiya3540/Next-Word-Predictor-LSTM.git
cd Next-Word-Predictor-LSTM

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Download corpus
python data/download_corpus.py
```

---

## 🚀 Usage

### Train the model
```bash
python train_now.py
```

### Run the app
```bash
streamlit run streamlit_app.py
```
Open: **http://localhost:8501**

### Run tests
```bash
pytest tests/ -v
# Expected: 66 passed ✅
```

---

## 📊 Results

| Metric | Alice (26K words) | 3 Books (225K words) |
|---|---|---|
| Train Accuracy | ~45% | ~65% |
| Val Accuracy | ~15% | ~40% |
| **Test Accuracy** | **~12%** | **~35%** |
| **Top-3 Accuracy** | **~35%** | **~60%** |
| Perplexity | ~90 | ~45 |

> Top-3 accuracy is the key metric for autocomplete — correct word appears in top-3 predictions.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| NLP | Custom LSTM Pipeline |
| Data | NumPy, Scikit-learn |
| Visualization | Matplotlib |
| Web App | Streamlit |
| Testing | Pytest (66 tests) |
| Version Control | Git + GitHub |

---

## 🔮 Future Improvements

| Priority | Feature |
|---|---|
| 🔴 High | Beam Search decoder |
| 🔴 High | Fine-tune GPT-2 / DistilGPT-2 |
| 🟡 Medium | Temperature sampling control |
| 🟡 Medium | Deploy on Hugging Face Spaces |
| 🟡 Medium | Hindi + English multilingual support |
| 🟢 Low | Docker container |
| 🟢 Low | REST API endpoint |

---

## 👩‍💻 Author

**Rajiya**

📌 BCA Final Year — JB Knowledge Park, Faridabad (MDU)
📌 Data Science & AI Certified — Ducat School of AI
📌 3+ Years BFSI Experience — SBI, Paisabazaar, Unifinz Capital

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rajiya--khatoon-blue?logo=linkedin)](https://linkedin.com/in/rajiya-khatoon-b4434327a)
[![GitHub](https://img.shields.io/badge/GitHub-Rajiya3540-black?logo=github)](https://github.com/Rajiya3540)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using TensorFlow and Streamlit*
