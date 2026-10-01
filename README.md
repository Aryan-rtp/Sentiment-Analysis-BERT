# 🎬 Sentiment Analysis using BERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Sentiment-Analysis-BERT/blob/main/Sentiment_Analysis_using_BERT.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-yellow)

Fine-tuning Google's **BERT** (`bert-base-uncased`) to classify IMDB movie reviews as **Positive** or **Negative**.

---

## 📌 Overview

Sentiment Analysis is a Natural Language Processing (NLP) task that identifies the emotional tone behind a piece of text. This project uses **transfer learning**: a BERT model pre-trained on large text corpora is fine-tuned on labelled movie reviews, achieving high accuracy with a relatively small amount of training data.

## 📂 Dataset

| Property | Details |
|---|---|
| Name | [IMDB Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) (`stanfordnlp/imdb`) |
| Total size | 50,000 reviews (25k train / 25k test) |
| Classes | 0 = Negative, 1 = Positive (balanced) |
| Subset used | 5,000 train / 2,000 test (for faster training) |

## 🧠 Model

| Property | Details |
|---|---|
| Base model | `bert-base-uncased` |
| Architecture | 12 Transformer encoder layers, 768 hidden size, 12 attention heads |
| Parameters | ~110 Million |
| Task head | `BertForSequenceClassification` (linear layer on the `[CLS]` token) |

## ⚙️ Workflow

```
Raw Text ──► Tokenization (WordPiece) ──► BERT Encoder ──► [CLS] Embedding ──► Classifier ──► Positive / Negative
```

1. **Load & explore data**: class distribution and review-length analysis
2. **Tokenize**: BERT tokenizer adds `[CLS]`/`[SEP]` tokens, truncates to 256 tokens, with dynamic padding
3. **Fine-tune**: Hugging Face `Trainer` API
4. **Evaluate**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, Loss curve
5. **Predict**: sentiment and confidence score on custom sentences

## 🔧 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 2 |
| Batch size | 16 (train) / 32 (eval) |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Warmup steps | 60 (~10%) |
| Max sequence length | 256 |
| Mixed precision | FP16 (on GPU) |

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~0.90 |
| Precision | ~0.90 |
| Recall | ~0.90 |
| F1-score | ~0.90 |

> Replace these with the exact values from your notebook run.

### Sample Predictions

| Review | Prediction |
|---|---|
| "This movie was an absolute masterpiece, I loved every minute of it!" | ✅ Positive |
| "Terrible acting and a boring plot. Complete waste of time." | ❌ Negative |
| "I would definitely recommend this film to my friends." | ✅ Positive |
| "Worst experience ever, I walked out halfway through." | ❌ Negative |

## 🛠️ Tech Stack

- **Python**
- **PyTorch**: deep learning framework
- **Hugging Face Transformers**: BERT model and Trainer
- **Hugging Face Datasets**: IMDB dataset loading
- **scikit-learn**: evaluation metrics
- **Matplotlib / Seaborn**: visualizations

## 🚀 How to Run

### Option 1: Google Colab (recommended)
1. Click the **Open in Colab** badge above.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**. Training takes about 10–15 minutes.

### Option 2: Local machine
```bash
git clone https://github.com/YOUR_USERNAME/Sentiment-Analysis-BERT.git
cd Sentiment-Analysis-BERT
pip install transformers datasets evaluate scikit-learn accelerate torch matplotlib seaborn pandas jupyter
jupyter notebook Sentiment_Analysis_using_BERT.ipynb
```
> A CUDA-enabled GPU is strongly recommended. Training on CPU will be very slow.

## 📁 Project Structure

```
Sentiment-Analysis-BERT/
├── Sentiment_Analysis_using_BERT.ipynb   # Main notebook
└── README.md                             # Project documentation
```

## 💡 Why BERT?

- **Bidirectional context**: reads text left-to-right and right-to-left at the same time, so it handles negation like *"not good"* correctly.
- **Transfer learning**: pre-trained on Wikipedia and BooksCorpus, so it needs only a small labelled dataset.
- **Self-attention**: learns which words matter most for each prediction.

## 🔮 Future Improvements

- Train on the full 25,000-review dataset
- Increase `max_length` to 512 tokens
- Compare against **DistilBERT** (faster) and **RoBERTa** (more accurate)
- Hyperparameter tuning (learning rate, epochs, batch size)
- Deploy as a web app using **Gradio** or **Streamlit**

## 👤 Author

**Aryan Pandey**
🔗 GitHub: [@Aryan-rtp](https://github.com/Aryan-rtp)

---

⭐ If you found this project helpful, consider giving it a star!
