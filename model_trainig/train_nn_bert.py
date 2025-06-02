import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from eval import mbti_accuracies
from pathlib import Path

# Load and preprocess data

# Load train/test sets with BERT embeddings + features
project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"

X_train_emb = pd.read_csv(processed_dir / "train_bert.csv", index_col=0)
X_test_emb = pd.read_csv(processed_dir / "test_bert.csv", index_col=0)

# Extract labels and remove non-feature columns
y_train = X_train_emb[['target_1', 'target_2', 'target_3', 'target_4']].values.astype(np.float32)
X_train = X_train_emb.drop(columns=['target_1', 'target_2', 'target_3', 'target_4', 'type', 'posts'])

y_test = X_test_emb[['target_1', 'target_2', 'target_3', 'target_4']].values.astype(np.float32)
X_test = X_test_emb.drop(columns=['target_1', 'target_2', 'target_3', 'target_4', 'type', 'posts'])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Convert features to NumPy
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Custom Dataset

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


# Create Dataloaders

batch_size = 256

train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Define  Neural Network

class TheModelClass(nn.Module):
    def __init__(self, input_size):
        super(TheModelClass, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.BN1 = nn.BatchNorm1d(128)
        self.Drout1 = nn.Dropout(p=0.6)

        self.fc2 = nn.Linear(128, 128)
        self.BN2 = nn.BatchNorm1d(128)
        self.Drout2 = nn.Dropout(p=0.6)

        self.fc3 = nn.Linear(128, 64)
        self.BN3 = nn.BatchNorm1d(64)
        self.Drout3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(64, 32)
        self.BN4 = nn.BatchNorm1d(32)

        self.fc5 = nn.Linear(32, 4)

        self.activation_relu = nn.ReLU()
        self.activation_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation_relu(self.fc1(x))
        x = self.Drout1(x)
        x = self.BN1(x)

        x = self.activation_relu(self.fc2(x))
        x = self.Drout2(x)
        x = self.BN2(x)

        x = self.activation_relu(self.fc3(x))
        x = self.Drout3(x)
        x = self.BN3(x)

        x = self.activation_relu(self.fc4(x))
        x = self.BN4(x)

        x = self.fc5(x)
        x = self.activation_sigmoid(x)

        return x

# Training Setup

input_size = X_train.shape[1]
model = TheModelClass(input_size).to(device)

learning_rate = 0.01
epochs = 100

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()


# Training Loop

for epoch in range(epochs):
    model.train()
    losses = []

    for X_batch, y_batch in train_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    train_loss = np.mean(losses)

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_losses = []
        for X_val, y_val in test_dataloader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            pred_val = model(X_val)
            loss_val = criterion(pred_val, y_val)
            val_losses.append(loss_val.item())
        mean_loss_valid = np.mean(val_losses)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Validation Loss: {mean_loss_valid:.4f}")

# Final Evaluation

with torch.no_grad():
    all_preds = []
    for X_batch, _ in test_dataloader:
        X_batch = X_batch.to(device)
        y_preds = model(X_batch)
        all_preds.extend(y_preds.cpu().numpy().tolist())

# Evaluate using MBTI accuracy metrics
mbti_accuracies(y_test, all_preds)
