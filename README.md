# Disaster Tweets: Real Disaster Tweet Classification

End-to-end NLP project for classifying whether a tweet refers to a real disaster or not. The project uses the Kaggle **Natural Language Processing with Disaster Tweets** dataset and compares multiple recurrent neural network architectures.

## Project Overview
This repository demonstrates a complete text-classification workflow:
- dataset inspection and exploratory analysis
- text cleaning and preprocessing
- tokenization, vocabulary building, and sequence padding
- recurrent neural network modeling in PyTorch
- model comparison and results reporting

## Problem Statement
Tweets often use disaster-related words figuratively, sarcastically, or in news-like contexts. The goal is to build a binary classifier that distinguishes **real disaster tweets** from **non-disaster tweets**.

## Dataset
Source: Kaggle - *Natural Language Processing with Disaster Tweets*

Expected files in `data/raw/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

Observed dataset sizes from the notebook:
- training rows: **7,613**
- test rows: **3,263**

## Tech Stack
- Python
- Pandas / NumPy
- Matplotlib / Seaborn
- NLTK
- PyTorch
- Scikit-learn
- Jupyter Notebook

## Methodology
1. Load and inspect the Kaggle training and test sets
2. Explore class balance, tweet length, keywords, and locations
3. Clean text by removing URLs, punctuation, mentions, and noisy tokens
4. Remove stopwords and lemmatize tokens
5. Convert processed text into padded integer sequences
6. Train and compare:
   - baseline BiLSTM
   - deeper stacked BiLSTM
   - BiGRU
7. Evaluate on a validation split using F1, accuracy, precision, and recall

## Key Results
From the notebook's recorded runs:
- **Baseline BiLSTM** — F1: `0.6958`, Accuracy: `0.7761`
- **Deeper BiLSTM** — F1: `0.7268`, Accuracy: `0.7853`
- **BiGRU** — F1: `0.7270`, Accuracy: `0.7859`

**Best model:** BiGRU on validation F1 and accuracy.

More detail is available in [`RESULTS.md`](RESULTS.md).

## What Was Fixed
This cleanup addressed several concrete issues in the original notebook:
- added robust repository-relative dataset paths
- added automatic NLTK resource downloads for fresh environments
- fixed a mislabeled locations plot title
- fixed missing `plt.show()` calls
- removed test-set leakage from vocabulary construction by fitting the vocabulary on training text only
- redirected model checkpoint files into `reports/models/`

## Repository Structure
```text
Disaster-Tweets/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── disaster_tweets_nlp.ipynb
├── reports/
│   ├── figures/
│   └── models/
├── src/
├── DATA_DICTIONARY.md
├── RESULTS.md
├── requirements.txt
└── README.md
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/lhimmelspach/Disaster-Tweets.git
   cd Disaster-Tweets
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the Kaggle dataset and place the CSV files in `data/raw/`.
5. Launch Jupyter Lab:
   ```bash
   jupyter lab
   ```
6. Open `notebooks/disaster_tweets_nlp.ipynb` and run the notebook top to bottom.

## Limitations
- Only the tweet text is used for training; `keyword` and `location` are explored but not modeled.
- The project currently evaluates on a single validation split rather than cross-validation.
- Results are respectable but not state of the art.

## Future Improvements
- add TF-IDF + linear model baselines
- incorporate `keyword` and `location` features
- try pretrained embeddings and transformer models
- export a reproducible prediction pipeline outside the notebook

## Author
**Luke Himmelspach**  
GitHub: https://github.com/lhimmelspach
