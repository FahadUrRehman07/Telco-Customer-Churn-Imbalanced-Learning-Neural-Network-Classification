from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .baseline import build_baseline_model
from .preprocessing import normalize_telco_frame, split_features_target

@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def evaluate_thresholds(frame: pd.DataFrame, thresholds=(0.30, 0.40, 0.50, 0.60, 0.70), test_size: float = 0.20, random_state: int = 42):
    """Train the ANN once and compare probability cutoffs on the test split."""
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("thresholds must be strictly between 0 and 1")
    data = normalize_telco_frame(frame)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    model = build_baseline_model(X_train)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    results = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        results.append(ThresholdResult(threshold, accuracy_score(y_test, predictions), precision_score(y_test, predictions, zero_division=0), recall_score(y_test, predictions, zero_division=0), f1_score(y_test, predictions, zero_division=0), roc_auc_score(y_test, probabilities), int(tn), int(fp), int(fn), int(tp)))
    return results
