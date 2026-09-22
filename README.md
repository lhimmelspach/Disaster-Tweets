# Disaster Tweets: Reproducible NLP Classification

> **Portfolio project:** a notebook-based NLP workflow for classifying tweets as disaster-related or not, with honest baselines, reproducible training, and explicit error analysis.

This repository analyzes the **Natural Language Processing with Disaster Tweets** dataset and compares simple and neural text-classification approaches. The project is intentionally compact, but the workflow is now technically correct:

- processed train/test text is assigned back **positionally** instead of through mismatched pandas indexes
- learned artifacts are fit on the **training split only**, avoiding avoidable validation/test leakage
- random seeds are set for Python, NumPy, and PyTorch
- the notebook handles required NLTK resources and missing local dataset files clearly
- validation reporting includes **accuracy, precision, recall, F1, and confusion matrices**
- the notebook prints representative **false positives** and **false negatives** for interpretation

## Repository contents

```text
Disaster-Tweets/
├── NaturalLanguageProcessing (1).ipynb
├── README.md
└── requirements.txt
```

## What the notebook demonstrates

- exploratory analysis of short, noisy social-media text
- deterministic text cleaning and lemmatization with NLTK
- stratified train/validation splitting
- a **majority-class baseline**
- a **TF-IDF + Logistic Regression** baseline
- a **PyTorch BiLSTM** with early stopping and gradient clipping
- metric-based model comparison on the validation split
- qualitative error analysis for ambiguous language

## Data

The repository does **not** include the Kaggle competition files. To run the full workflow locally, place these files in the repository root:

```text
train.csv
test.csv
```

Those files are external competition data and should not be committed.

## Corrected workflow

### 1. Preprocessing

The notebook lowercases text, removes URLs/HTML/mentions/punctuation/numeric tokens, keeps hashtag words by removing only the `#` symbol, removes English stop words, and lemmatizes remaining tokens.

To preserve the original educational approach while fixing the bug, train and test text are still preprocessed together for deterministic cleaning, but the processed text is assigned back like this:

```python
train_df["processed_text"] = combined_df.iloc[: len(train_df)]["processed_text"].to_numpy()
test_df["processed_text"] = combined_df.iloc[len(train_df) :]["processed_text"].to_numpy()
```

Using `.to_numpy()` forces **positional assignment**, which prevents the earlier all-padding test-sequence failure caused by pandas index alignment.

### 2. Leakage prevention

The notebook now learns all fitted text artifacts from the **training split only**:

- the TF-IDF vectorizer is fit on the training split and used to transform validation/test text
- the BiLSTM vocabulary is built from the training split and used to encode validation/test text
- the maximum sequence length is derived from the training split

This keeps validation metrics honest and avoids using unlabeled test text to define the feature space.

### 3. Reproducibility

The notebook sets seeds for:

- Python `random`
- NumPy
- PyTorch
- train/validation splitting
- shuffled PyTorch dataloaders

It also attempts deterministic PyTorch behavior where practical and downloads required NLTK corpora if they are missing locally. If corpus downloads are blocked in a restricted environment, the notebook falls back gracefully by using scikit-learn English stop words and skipping lemmatization rather than crashing.

## Results and model comparison

The notebook is configured to compare:

1. **Majority class baseline**
2. **TF-IDF + Logistic Regression**
3. **BiLSTM**

For each model, the notebook reports:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

### Important note about results

No numeric results are claimed in this README because the dataset files were **not present in the repository clone used for this update**. The notebook will calculate and print the real validation metrics locally once `train.csv` and `test.csv` are added.

That keeps the repository honest: the workflow is corrected and ready to run, but no unverified scores are advertised.

## Error analysis

The notebook now prints representative false positives and false negatives from the best validation model among the main learned baselines.

This matters because disaster language is ambiguous:

- **false positives** often involve figurative language such as “this exam was a disaster” or “my phone is on fire”
- **false negatives** often require context that is not obvious from keywords alone

For a hiring manager, this is a more useful signal than a single metric because it shows model judgment, limitations, and interpretation.

## Environment assumptions

This update was prepared in a **Python 3.12** environment. Install the dependencies in `requirements.txt` and run the notebook from the repository root so relative dataset paths resolve correctly.

## Setup

```bash
git clone https://github.com/lhimmelspach/Disaster-Tweets.git
cd Disaster-Tweets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then place `train.csv` and `test.csv` in the repository root and launch Jupyter:

```bash
jupyter lab
```

Open `NaturalLanguageProcessing (1).ipynb` and run the cells from top to bottom.

## Limitations

- the project is still notebook-first rather than a packaged training pipeline
- the dataset is small for a neural model trained from scratch
- tweets are short, noisy, and often ambiguous without external context
- validation performance can vary across random seeds and splits
- the notebook does not commit benchmark outputs because the external dataset is intentionally absent

## Why this is portfolio-worthy

This repository now presents the project the way a hiring manager would want to see it:

- a clear problem statement
- a technically correct preprocessing pipeline
- an honest baseline before a neural model
- reproducibility steps
- interpretable validation reporting
- explicit discussion of ambiguity and model errors

## Author

**Luke Himmelspach**  
[GitHub](https://github.com/lhimmelspach)
