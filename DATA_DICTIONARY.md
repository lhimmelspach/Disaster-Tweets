# Disaster Tweets - Data Dictionary

This document describes the key raw and engineered fields used in the project.

## Raw Dataset Fields

### `id`
- **Type**: Integer
- **Description**: Unique tweet identifier
- **Used for modeling**: No

### `keyword`
- **Type**: String / categorical
- **Description**: Disaster-related keyword associated with the tweet
- **Missing values**: Present in both train and test sets
- **Used for modeling**: Not in the current notebook models

### `location`
- **Type**: String / categorical
- **Description**: User-provided location text
- **Missing values**: High missingness
- **Used for modeling**: Not in the current notebook models

### `text`
- **Type**: String
- **Description**: Raw tweet text
- **Used for modeling**: Yes, after preprocessing

### `target`
- **Type**: Integer (binary)
- **Values**:
  - `1` = real disaster tweet
  - `0` = non-disaster tweet
- **Used for modeling**: Yes (training labels only)

## Engineered Fields in the Notebook

### `text_len`
- **Type**: Integer
- **Description**: Character length of each tweet
- **Purpose**: Exploratory analysis

### `cleaned_text`
- **Type**: String
- **Description**: Tweet text after lowercasing and noise removal
- **Transformations**:
  - remove URLs
  - remove HTML tags
  - remove punctuation
  - remove mentions
  - drop words containing digits
  - strip the `#` symbol while keeping hashtag words

### `processed_text`
- **Type**: String
- **Description**: Cleaned text after stopword removal and lemmatization
- **Purpose**: Primary text feature used for sequence modeling

### `processed_text_len`
- **Type**: Integer
- **Description**: Token count of `processed_text`
- **Purpose**: Determines the padding/truncation length threshold

## Sequence Representation

### Vocabulary
- Built from the **training processed text only**
- Includes special tokens:
  - `<pad>` → `0`
  - `<unk>` → `1`

### Padded sequence
- **Type**: List of integers
- **Description**: Integer-encoded version of `processed_text` padded/truncated to a fixed maximum length

## Data Notes
- Training set rows: **7,613**
- Test set rows: **3,263**
- `keyword` and `location` are useful exploratory fields but are not currently modeled because of missingness and high cardinality.
