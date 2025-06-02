# Predicting Personality Profiles Through Social Media Content

This project aims to classify MBTI personality types based on Reddit user posts using various NLP models, including TF-IDF + XGBoost, BERT + NN, and LLM fine-tuning with Mistral-7B.


All Python scripts:
------------------
- `scrape_reddit.py`
- `data_processing/process_data.py`
- `model_training/train_tfidf_xgboost.py`
- `data_processing/extract_bert_embeddings.py`
- `model_training/train_nn_bert.py`
- `model_training/fine_tune_bert.py` 
- `model_training/fine_tune_Mistral.py`



Features:
---------
Implemented steps of this project
- Reddit scraping for MBTI-labeled users (scrape_reddit)
- Text cleaning, lemmatization, emoji & URL removal (process_data)
- TF-IDF + XGBoost multi-label classification (train_tfidf_xgboost)
- BERT embedding + FFNN model (extract_bert_embeddings, train_nn_bert)
- BERT fine-tuning (fine_tune_bert)
- Mistral-7B LoRA fine-tuning for classification (fine_tune_Mistral)

## Pipelines

> **Note:** Due to memory limitations, the data sources are not included in this repository.  
> If you would like access to the data to reproduce the results, please contact me.

Run each stage in order:

1. `python scrape_reddit.py`  
2. `python data_processing/process_data.py`  
3. `python model_trainig/train_tfidf_xgboost.py`  
4. `python data_processing/extract_bert_embeddings.py`  
5. `python model_trainig/train_nn_bert.py`  
6. `python model_trainig/fine_tune_bert.py`  
7. `python model_trainig/fine_tune_Mistral.py`

## Requirements

- pandas==2.2.3  
- numpy==1.26.4  
- tqdm==4.66.5  
- spacy==3.8.5  
- nltk==3.9.1  
- textblob==0.19.0  
- emoji==2.14.1  
- scikit-learn==1.5.2  
- xgboost==1.7.6  
- torch==2.4.1  
- transformers==4.51.3  
- datasets==3.4.1  
- peft==0.15.2  
- accelerate==1.5.2  
- bitsandbytes==0.45.5  
- praw==7.8.1  
