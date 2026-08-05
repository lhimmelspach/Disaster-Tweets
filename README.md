# Disaster Tweets Classification

Natural language processing (NLP) project to classify whether a tweet is about a real disaster.

## Project Overview
This project builds a binary text classifier to distinguish:
- **1** = tweet references a real disaster  
- **0** = tweet is not about a real disaster

It covers text preprocessing, feature extraction, model training, and evaluation.

## Problem Statement
Social media can provide real-time situational signals during emergencies, but raw streams are noisy.  
The goal is to classify disaster-related tweets accurately to support triage and monitoring workflows.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK / spaCy (if used)
- Jupyter Notebook

## Methodology
1. Loaded and inspected labeled tweet data  
2. Cleaned text (URLs, punctuation, casing, stopwords as appropriate)  
3. Converted text to numerical features (e.g., TF-IDF)  
4. Trained baseline and advanced classifiers  
5. Evaluated using classification metrics and error analysis

### Key Findings
- TF-IDF + linear classifier provided strong baseline performance.
- False positives often came from figurative language.
- Class imbalance handling improved recall for disaster tweets.

## Repository Structure
```text
Disaster-Tweets/
├── data/
├── notebooks/
├── src/
├── reports/figures/
├── requirements.txt
└── README.md
```

## How to Run
1. Clone:
   ```bash
   git clone https://github.com/lhimmelspach/Disaster-Tweets.git
   cd Disaster-Tweets
   ```
2. Create environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run notebooks:
   ```bash
   jupyter lab
   ```

## Limitations
- Tweets are short and context-poor, which increases ambiguity.
- Figurative language and sarcasm can reduce precision.
- Performance may degrade on events or language styles not
