import os
import time
import math
import torch
import numpy as np
import pandas as pd
from torch import cuda, nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from eval import mbti_accuracies
from pathlib import Path

# Set device
device = 'cuda' if cuda.is_available() else 'cpu'

# Clear CUDA memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"
# Load and sample data (temporary sampling for faster testing)
train_data = pd.read_csv(processed_dir / "train_data.csv", index_col=0)
test_data = pd.read_csv(processed_dir / "test_data.csv", index_col=0)
# Extract labels and text data
y_train = train_data[['target_1', 'target_2', 'target_3', 'target_4']].values.astype(np.float32)
y_test = test_data[['target_1', 'target_2', 'target_3', 'target_4']].values.astype(np.float32)

labels_train = y_train.tolist()
labels_test = y_test.tolist()
texts_train = train_data['posts'].to_list()
texts_test = test_data['posts'].to_list()

# Tokenize input texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_token = tokenizer(texts_train, padding="max_length", truncation=True, max_length=500, return_tensors="pt")
test_token = tokenizer(texts_test, padding="max_length", truncation=True, max_length=500, return_tensors="pt")

class MultiLabelDataset(Dataset):
    """
    Dataset class for multi-label classification.
    Each sample includes BERT tokenized inputs and a vector of 4 binary targets.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Prepare dataloaders
train_dataset = MultiLabelDataset(train_token, labels_train)
test_dataset = MultiLabelDataset(test_token, labels_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Load BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4,
    problem_type="multi_label_classification"
)
model.to(device)

# Freeze all BERT layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 3 encoder layers and classifier head
for layer in model.bert.encoder.layer[-3:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

# Loss, optimizer, and learning rate scheduler
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
model.train()
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None)
        )
        logits = outputs.logits
        loss = loss_fn(logits, batch['labels'])
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Loss: {loss.item()}")

# Evaluation loop
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_dataloader:
        # Remove labels for prediction
        if 'labels' in batch:
            batch.pop('labels')
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        all_preds.extend(probs.cpu().tolist())

# Evaluate predictions using custom MBTI metrics
mbti_accuracies(y_test, all_preds)
