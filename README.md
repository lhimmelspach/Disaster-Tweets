# Disaster Tweets: NLP Classification with PyTorch

> **Portfolio project:** classifying short social-media posts as disaster-related or not disaster-related.

This project explores how natural-language processing can support the triage of noisy, ambiguous social-media text during emergencies. It includes exploratory data analysis, text normalization, vocabulary construction, sequence padding, and a bidirectional LSTM classifier implemented in PyTorch.

## Executive summary

| Area | Details |
|---|---|
| **Task** | Binary text classification: disaster tweet vs. non-disaster tweet |
| **Training data** | 7,613 labeled tweets |
| **Test data** | 3,263 unlabeled tweets |
| **Target balance** | 42.97% disaster / 57.03% non-disaster tweets |
| **Primary model** | Bidirectional LSTM with learned word embeddings |
| **Frameworks** | Python, pandas, scikit-learn, NLTK, PyTorch |
| **Evaluation focus** | Precision, recall, F1 score, accuracy, and error analysis |

## Why this problem matters

Emergency-response teams may need to identify useful signals from large volumes of informal, short, and context-poor messages. A practical classifier must do more than achieve a high accuracy score: it should make the tradeoff between missed disaster tweets and false alarms explicit, remain reproducible, and be evaluated on examples where language is ambiguous.

## What this project demonstrates

- Translating an open-ended NLP problem into a measurable classification task
- Inspecting data quality, class balance, and missing values before modeling
- Cleaning informal text while preserving potentially meaningful hashtag terms
- Building a vocabulary and converting variable-length text into padded sequences
- Implementing a trainable neural network in PyTorch rather than relying only on a high-level estimator
- Using stratified validation splitting and binary cross-entropy with logits
- Diagnosing preprocessing and inference failures instead of silently reporting invalid predictions

## Dataset

The project uses the **Natural Language Processing with Disaster Tweets** dataset. Each record contains:

- `text`: the tweet content
- `keyword`: an extracted disaster-related keyword, when available
- `location`: the reported location, when available
- `target`: `1` for a real disaster tweet and `0` otherwise

The training data contains missing values in `keyword` and `location`, while `text` and `target` are complete. The notebook currently focuses primarily on the cleaned tweet text.

The dataset files are intentionally not committed to this repository. To run the notebook, place the competition files in the repository root:

```text
train.csv
 test.csv
```

## Exploratory analysis

The notebook investigates:

- Target-class distribution
- Tweet-length distributions by class
- Frequent keywords and locations
- Frequent words after preprocessing
- Missing values and basic dataset structure

The analysis shows that disaster tweets are somewhat longer on average than non-disaster tweets in this sample, while the target distribution is moderately imbalanced rather than severely skewed.

## Text preprocessing

The current preprocessing workflow:

1. Converts text to lowercase
2. Removes URLs, HTML tags, punctuation, newline characters, and mentions
3. Removes words containing numbers
4. Removes the `#` symbol while retaining the hashtag word
5. Removes English stop words
6. Applies WordNet lemmatization
7. Builds a vocabulary with padding and unknown tokens
8. Truncates or pads sequences to the 95th-percentile sequence length

This creates a compact input representation for the neural network while retaining the semantic content of hashtags such as `#wildfires`.

## Model architecture

The notebook implements a PyTorch `BiLSTMClassifier` with:

- 100-dimensional learned word embeddings
- 128 hidden units in a bidirectional LSTM
- Global max pooling over sequence outputs
- A 64-unit fully connected layer
- Dropout regularization with probability 0.3
- A single binary output logit
- `BCEWithLogitsLoss` and the Adam optimizer
- Gradient clipping to reduce the risk of exploding gradients

The configured model contains approximately **1.8 million trainable parameters**.

## Results and validation status

Final benchmark metrics are intentionally **not reported yet** because the current notebook contains a preprocessing/inference issue that must be fixed before its predictions can be trusted.

During the notebook run, a diagnostic check reported that every test sequence was identical and contained only padding-token IDs. This means the model was not receiving the actual test text at inference time. Reporting a test score or submission result before correcting this would be misleading.

### Known issue to fix

The train/test processed-text assignment is performed using pandas index alignment after slicing `combined_df`. Because the test slice retains indexes beginning at the training-set length, assigning it to `test_df` can produce missing values that are later converted into empty strings. The result is an all-padding test input.

The assignment should be made positionally, for example:

```python
train_df["processed_text"] = combined_df.iloc[: len(train_df)]["processed_text"].to_numpy()
test_df["processed_text"] = combined_df.iloc[len(train_df) :]["processed_text"].to_numpy()
```

After fixing this issue, the project should report:

- A majority-class baseline
- A simple TF-IDF + logistic regression baseline
- BiLSTM validation accuracy, precision, recall, and F1
- A confusion matrix
- Representative false positives and false negatives
- Results across at least one additional random seed or validation split

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/lhimmelspach/Disaster-Tweets.git
cd Disaster-Tweets
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk torch jupyter
```

The notebook also requires the NLTK English stop-word and WordNet resources. In a Python session, run:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

### 4. Add the dataset files

Place `train.csv` and `test.csv` in the repository root. These files are not included because they are external competition data.

### 5. Run the notebook

```bash
jupyter lab
```

Open `NaturalLanguageProcessing (1).ipynb` and run the cells from top to bottom. The notebook should be updated to fix the test-sequence issue before treating its final predictions as valid.

## Recommended next improvements

1. Fix the positional train/test assignment described above.
2. Build the vocabulary from the training split only to avoid validation/test information leakage.
3. Add a reproducible TF-IDF baseline before comparing against the BiLSTM.
4. Save validation metrics and plots under `reports/` rather than only displaying them in the notebook.
5. Add early stopping, checkpointing, and a random seed for reproducible training.
6. Evaluate precision and recall separately because false negatives may be especially costly in an emergency-triage setting.
7. Add error analysis for sarcasm, figurative uses of words such as “fire,” duplicated tweets, and ambiguous news references.
8. Refactor reusable preprocessing and modeling code into `src/`, with a `requirements.txt` file and a small test suite.

## Repository contents

```text
Disaster-Tweets/
├── NaturalLanguageProcessing (1).ipynb  # EDA, preprocessing, modeling, diagnostics
└── README.md                             # Project documentation
```

## Limitations

- The dataset is relatively small for training a neural language model from scratch.
- Tweets are short, noisy, and often ambiguous without external context.
- The labels may reflect annotator judgment rather than an objective definition of a disaster.
- Validation performance may not generalize to future events, regions, or writing styles.
- The current repository is notebook-only and does not yet provide a fully automated training or inference pipeline.

## Author

**Luke Himmelspach**  
[GitHub](https://github.com/lhimmelspach)
