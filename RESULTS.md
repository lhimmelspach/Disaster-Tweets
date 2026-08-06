# Disaster Tweets - Results Summary

## Executive Summary
This project compares three recurrent neural network architectures for classifying disaster-related tweets. The best recorded validation performance came from the **BiGRU** model, which slightly outperformed both BiLSTM variants.

## Validation Split
- Training split: **6,090** rows
- Validation split: **1,523** rows
- Sequence length: **15** tokens at the 95th percentile
- Vocabulary size: **15,435** tokens in the original recorded run

## Model Comparison

| Model | Validation F1 | Validation Accuracy | Notes |
|---|---:|---:|---|
| Baseline BiLSTM | 0.6958 | 0.7761 | Strong precision, weaker recall on disaster tweets |
| Deeper BiLSTM | 0.7268 | 0.7853 | Improved balance over the baseline |
| BiGRU | 0.7270 | 0.7859 | Best overall recorded validation metrics |

## Detailed Baseline Metrics
From the notebook's baseline classification report:
- Precision: **0.8351**
- Recall: **0.5963**
- F1-score: **0.6958**
- Accuracy: **0.7761**

## Main Observations
1. The recurrent models cluster closely together, so architecture changes alone are not producing dramatic gains.
2. The baseline favors precision over recall, meaning it misses some true disaster tweets.
3. Better gains will likely come from improved feature representations, lighter-touch preprocessing, or stronger baselines such as TF-IDF + linear models and transformers.

## Recommendations
- Benchmark against TF-IDF with Logistic Regression or Linear SVM.
- Add `keyword` and `location` features instead of relying only on text.
- Try pretrained embeddings or transformer models such as BERT.
- Consider cross-validation to reduce dependence on a single random split.
