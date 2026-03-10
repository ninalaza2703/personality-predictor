import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from eval import mbti_accuracies
from pathlib import Path

# Load preprocessed data
project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"

train_data = pd.read_csv(processed_dir / "train_data.csv")
test_data = pd.read_csv(processed_dir / "test_data.csv")

X_train = train_data['posts']
y_train = train_data[['target_1', 'target_2', 'target_3', 'target_4']]

X_test = test_data['posts']
y_test = test_data[['target_1', 'target_2', 'target_3', 'target_4']]

# TF-IDF Vectorization

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

train_post = vectorizer.transform(X_train)
test_post = vectorizer.transform(X_test)

# Multi-output XGBoost Classifier

xgb = XGBClassifier(tree_method='auto')
multi_target_model = MultiOutputClassifier(xgb)
multi_target_model.fit(train_post, y_train)


# Evaluation

y_pred = multi_target_model.predict(test_post)
y_test = np.array(y_test)

mbti_accuracies(y_test, y_pred)
