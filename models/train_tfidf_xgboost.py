import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion


DIM_NAMES = ["target_1", "target_2", "target_3", "target_4"]
TEXT_COL = "posts"
LABEL_COLS = ["target_1", "target_2", "target_3", "target_4"]

MBTI_TYPES = [
    "INTP", "INTJ", "INFJ", "INFP", "ENTP", "ENTJ", "ENFJ", "ENFP",
    "ISTP", "ISTJ", "ISFJ", "ISFP", "ESTP", "ESTJ", "ESFJ", "ESFP"
]
pat_mbti = re.compile(r"\b(" + "|".join(MBTI_TYPES) + r")\b", re.IGNORECASE)


def compute_fulltype_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    correct = (y_true == y_pred).sum(axis=1)
    return {
        "Full MBTI Type Accuracy (All 4 correct)": float((correct == 4).mean()),
        "At least 3 correct": float((correct >= 3).mean()),
        "At least 2 correct": float((correct >= 2).mean()),
        "At least 1 correct": float((correct >= 1).mean()),
    }


def per_dim_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(DIM_NAMES):
        rows.append({
            "dimension": name,
            "macro_f1": f1_score(y_true[:, k], y_pred[:, k], average="macro"),
            "balanced_acc": balanced_accuracy_score(y_true[:, k], y_pred[:, k]),
        })
    return pd.DataFrame(rows).set_index("dimension")


def tune_thresholds(probs_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    thrs = np.zeros(y_val.shape[1], dtype=float)
    for k in range(y_val.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            pred = (probs_val[:, k] >= t).astype(int)
            f1 = f1_score(y_val[:, k], pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thrs[k] = best_t
    return thrs


def train_predict_logreg(X_tr, y_tr, X_va, X_te):
    val_probs = np.zeros((X_va.shape[0], y_tr.shape[1]), dtype=float)
    test_probs = np.zeros((X_te.shape[0], y_tr.shape[1]), dtype=float)

    for k in range(y_tr.shape[1]):
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
        clf.fit(X_tr, y_tr[:, k])
        val_probs[:, k] = clf.predict_proba(X_va)[:, 1]
        test_probs[:, k] = clf.predict_proba(X_te)[:, 1]

    return val_probs, test_probs


def main():
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    train_df = pd.read_csv(processed_dir / "train_data.csv")
    test_df = pd.read_csv(processed_dir / "test_data.csv")

    y_train_full = train_df[LABEL_COLS].astype(int).values
    y_test = test_df[LABEL_COLS].astype(int).values

    tr_df, va_df, y_tr, y_va = train_test_split(
        train_df,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=train_df["target_1"],
    )

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )

    tfidf = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
    ])

    tfidf.fit(tr_df[TEXT_COL])

    X_tr = tfidf.transform(tr_df[TEXT_COL])
    X_va = tfidf.transform(va_df[TEXT_COL])
    X_te = tfidf.transform(test_df[TEXT_COL])

    val_probs, test_probs = train_predict_logreg(X_tr, y_tr, X_va, X_te)

    thresholds = tune_thresholds(val_probs, y_va)
    y_pred = (test_probs >= thresholds).astype(int)

    dim_results = per_dim_metrics(y_test, y_pred)
    fulltype_results = compute_fulltype_metrics(y_test, y_pred)

    print("\nPer-dimension metrics:")
    print(dim_results)

    print("\nFull-type metrics:")
    for metric_name, metric_value in fulltype_results.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nChosen thresholds:")
    for dim_name, thr in zip(DIM_NAMES, thresholds):
        print(f"{dim_name}: {thr:.2f}")


if __name__ == "__main__":
    main()