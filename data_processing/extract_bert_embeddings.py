import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"
train_data = pd.read_csv('data/processed/train_data.csv')
test_data = pd.read_csv('data/processed/test_data.csv')

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

model = model.to(device)

# Free up GPU memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


def get_embeddings(X, batch_size=64):
    """
    Generate BERT embeddings for input text data.

    Args:
        X (pd.Series): A series of text entries.
        batch_size (int): Number of samples per batch for inference.

    Returns:
        list: List of 768-dimensional embeddings (CLS token) for each input.
    """
    all_pred_list = []
    n_iters = math.ceil(len(X) / batch_size)

    for i in tqdm(range(n_iters)):
        batch_texts = X.iloc[i * batch_size:(i + 1) * batch_size].to_list()
        tok = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(
                input_ids=tok['input_ids'].to(device),
                token_type_ids=tok['token_type_ids'].to(device),
                attention_mask=tok['attention_mask'].to(device)
            )

        # Extract CLS token embedding (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_pred_list.extend(cls_embeddings.tolist())

    return all_pred_list


# Generate BERT embeddings
bert_emb_train = get_embeddings(train_data['posts'])
bert_emb_test = get_embeddings(test_data['posts'])

# Convert embeddings to DataFrames
emb_train = pd.DataFrame(bert_emb_train)
emb_test = pd.DataFrame(bert_emb_test)

# Concatenate embeddings with original data
train_data = pd.concat([emb_train, train_data], axis=1)
test_data = pd.concat([emb_test, test_data], axis=1)

# Prepare training features and multi-label targets
X_train_emb = train_data.iloc[:, :768]
y_train_emb = train_data[['target_1', 'target_2', 'target_3', 'target_4']]

# Train multi-label XGBoost model
xgb = XGBClassifier()
multi_target_model = MultiOutputClassifier(xgb)
multi_target_model.fit(X_train_emb, y_train_emb)

# Get top 20 most important features per label
top_20_per_label_sets = []
feature_names = X_train_emb.columns

for estimator in multi_target_model.estimators_:
    importances = estimator.feature_importances_
    top_features = pd.Series(importances, index=feature_names).nlargest(20).index.tolist()
    top_20_per_label_sets.extend(top_features)

# Add additional metadata and targets to keep
top_20_per_label_sets.extend([
    'type', 'posts', 'avg_word_length', 'lexical_diversity',
    'sentiment_polarity', 'sentiment_subjectivity',
    'target_1', 'target_2', 'target_3', 'target_4'
])

# Select reduced features and save to CSV
train_data_bert = train_data[top_20_per_label_sets]
test_data_bert = test_data[top_20_per_label_sets]

project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"

# Ensure directory exists
processed_dir.mkdir(parents=True, exist_ok=True)

# Save BERT-processed data
train_data_bert.to_csv(processed_dir / "train_bert.csv", index=False)
test_data_bert.to_csv(processed_dir / "test_bert.csv", index=False)
