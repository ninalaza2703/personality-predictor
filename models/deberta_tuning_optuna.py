import random
import shutil
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

TEXT_COL = "posts"
LABEL_COLS = ["target_1", "target_2", "target_3", "target_4"]
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 512


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(df: pd.DataFrame, label_cols: list[str]) -> torch.Tensor:
    y = df[label_cols].astype(int).values
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos = np.clip(pos, 1, None)
    pos_weight = (neg / pos).astype(np.float32)
    return torch.tensor(pos_weight)


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[[TEXT_COL] + LABEL_COLS].reset_index(drop=True))


def fulltype_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    correct = (y_true == y_pred).sum(axis=1)
    return {
        "full_type_acc_all4": float((correct == 4).mean()),
        "at_least_3": float((correct >= 3).mean()),
        "at_least_2": float((correct >= 2).mean()),
        "at_least_1": float((correct >= 1).mean()),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    macro_f1s = []
    bal_accs = []
    for k in range(labels.shape[1]):
        macro_f1s.append(f1_score(labels[:, k], preds[:, k], average="macro"))
        bal_accs.append(balanced_accuracy_score(labels[:, k], preds[:, k]))

    out = {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "balanced_acc_mean": float(np.mean(bal_accs)),
    }
    out.update(fulltype_metrics(labels, preds))
    return out


class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(model.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    set_seed(SEED)

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    outputs_dir = project_root / "outputs_deberta_optuna"
    model_dir = project_root / "models" / "deberta_mbti_finetuned_optuna_best"

    train_df = pd.read_csv(processed_dir / "train_data.csv")
    test_df = pd.read_csv(processed_dir / "test_data.csv")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[TEXT_COL] = train_df[TEXT_COL].astype(str)
    test_df[TEXT_COL] = test_df[TEXT_COL].astype(str)

    tr_df, va_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
    )

    pos_weight = compute_pos_weight(tr_df, LABEL_COLS)

    ds_train = to_hf_dataset(tr_df)
    ds_val = to_hf_dataset(va_df)
    ds_test = to_hf_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LEN,
        )

    def add_labels(batch):
        labels = np.stack([batch[c] for c in LABEL_COLS], axis=1).astype(np.float32)
        batch["labels"] = labels
        return batch

    ds_train_tok = ds_train.map(tokenize_batch, batched=True)
    ds_val_tok = ds_val.map(tokenize_batch, batched=True)
    ds_test_tok = ds_test.map(tokenize_batch, batched=True)

    ds_train_tok = ds_train_tok.map(add_labels, batched=True)
    ds_val_tok = ds_val_tok.map(add_labels, batched=True)
    ds_test_tok = ds_test_tok.map(add_labels, batched=True)

    cols_to_remove = [TEXT_COL] + LABEL_COLS
    ds_train_tok = ds_train_tok.remove_columns(cols_to_remove)
    ds_val_tok = ds_val_tok.remove_columns(cols_to_remove)
    ds_test_tok = ds_test_tok.remove_columns(cols_to_remove)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
        wd = trial.suggest_float("weight_decay", 0.0, 0.1)
        warmup = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        grad_acc = trial.suggest_categorical("grad_accum", [1, 2, 4])

        trial_output_dir = outputs_dir / f"trial_{trial.number}"

        args = TrainingArguments(
            output_dir=str(trial_output_dir),
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            logging_steps=200,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1_mean",
            greater_is_better=True,
            learning_rate=lr,
            weight_decay=wd,
            warmup_ratio=warmup,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=grad_acc,
            num_train_epochs=1,
            fp16=True,
            seed=SEED,
            report_to="none",
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=4,
            problem_type="multi_label_classification",
        ).to(DEVICE)

        trainer = WeightedBCETrainer(
            pos_weight=pos_weight,
            model=model,
            args=args,
            train_dataset=ds_train_tok,
            eval_dataset=ds_val_tok,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        metrics = trainer.evaluate(ds_val_tok)

        shutil.rmtree(trial_output_dir, ignore_errors=True)

        return metrics["eval_macro_f1_mean"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    best = study.best_params

    final_output_dir = project_root / "outputs_deberta_optuna_best"

    args_final = TrainingArguments(
        output_dir=str(final_output_dir),
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1_mean",
        greater_is_better=True,
        learning_rate=best["learning_rate"],
        weight_decay=best["weight_decay"],
        warmup_ratio=best["warmup_ratio"],
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=best["grad_accum"],
        num_train_epochs=3,
        fp16=True,
        seed=SEED,
        report_to="none",
    )

    model_final = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        problem_type="multi_label_classification",
    ).to(DEVICE)

    trainer_final = WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model_final,
        args=args_final,
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    trainer_final.train()

    test_metrics = trainer_final.evaluate(ds_test_tok)
    print(test_metrics)

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer_final.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()