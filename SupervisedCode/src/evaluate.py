# evaluate.py
# This file contains functions for evaluating model performance and saving metrics.

import json
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    mae_ordinal: float
    confusion: list[list[int]]

def evaluate(model, X, y) -> Metrics:
    pred = model.predict(X)

    acc = float(accuracy_score(y, pred))
    mf1 = float(f1_score(y, pred, average="macro"))

    # Compute MAE in a shared encoded label space.
    # Fit on y first; if pred has unseen labels in this split, extend to union.
    y_np = np.asarray(y).ravel()
    pred_np = np.asarray(pred).ravel()
    le = LabelEncoder()
    y_true = le.fit_transform(y_np)
    try:
        y_pred = le.transform(pred_np)
    except ValueError:
        le.fit(np.concatenate([y_np, pred_np]))
        y_true = le.transform(y_np)
        y_pred = le.transform(pred_np)
    mae = float(np.mean(np.abs(y_true - y_pred)))

    cm = confusion_matrix(y, pred).tolist()

    return Metrics(accuracy=acc, macro_f1=mf1, mae_ordinal=mae, confusion=cm)

def save_metrics(metrics: Metrics, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics.__dict__, f, indent=2)
