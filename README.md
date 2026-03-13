# Personality Predictor

This project predicts **MBTI personality traits from Reddit text** using multiple machine learning and deep learning approaches.  
It includes a full pipeline for scraping Reddit data, building a labeled dataset from user flairs, preprocessing text, and training several models ranging from TF-IDF baselines to transformer-based models and LLM fine-tuning.

---

## Project Structure

```
personality-predictor/
├── data_pipeline/
│   ├── build_reddit_dataset.py
│   ├── extract_bert_embeddings.py
│   ├── process_data.py
│   ├── scrape_reddit.py
│   └── scrape_reddit_pullpush.py
│
├── models/
│   ├── deberta_tuning.py
│   ├── deberta_tuning_optuna.py
│   ├── eval.py
│   ├── fine_tune_bert.py
│   ├── fine_tune_Mistral.py
│   ├── train_nn_bert.py
│   ├── train_tfidf_xgboost.py
│   └── utils.py
│
├── notebooks/
│   ├── deberta_tuning.ipynb
│   ├── deberta_tuning_optuna.ipynb
│   └── tfidf_experiments.ipynb
│
├── README.md
└── requirements.txt
```


The **notebooks** directory contains exploratory experiments used during development.  
The **main reproducible pipeline** is implemented in the `data_pipeline` and `models` directories.

---

# Project Goal

The goal of this project is to **predict MBTI personality dimensions from Reddit user text**.

The model predicts the four MBTI dimensions:

| Dimension | Description |
|----------|-------------|
| `target_1` | Extroversion (E) vs Introversion (I) |
| `target_2` | Intuition (N) vs Sensing (S) |
| `target_3` | Thinking (T) vs Feeling (F) |
| `target_4` | Perceiving (P) vs Judging (J) |

MBTI labels are extracted from **Reddit user flairs**, and comments from each author are aggregated to build training samples.

---
## Pipeline

### 1. Scrape Reddit data

```bash
python data_pipeline/scrape_reddit_pullpush.py
```

### 2. Build MBTI dataset

```bash
python data_pipeline/build_reddit_dataset.py
```

### 3. Process data

```bash
python data_pipeline/process_data.py
```

### 4. Train models

```bash
python models/train_tfidf_xgboost.py
python models/train_nn_bert.py
python models/fine_tune_bert.py
python models/deberta_tuning.py
python models/deberta_tuning_optuna.py
python models/fine_tune_Mistral.py
```

---

## Installation

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
---

## Hardware

CPU is sufficient for preprocessing and classical models.

GPU is recommended for transformer and LLM training.  
Development experiments used an **NVIDIA RTX 3090**.

---

## Data

The Reddit dataset is not included in the repository, you can:

- run the scraping pipeline
- request the dataset from me
