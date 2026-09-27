"""Random over-sampling experiment for the Telco churn project."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from .baseline import build_baseline_model
from .preprocessing import normalize_telco_frame, split_features_target

@dataclass(frozen=True)
class OversamplingResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_rows_before: int
    train_rows_after: int

def oversample_training_data(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Duplicate minority rows until both classes have equal counts."""
    train = X_train.copy()
    train["__target__"] = y_train.to_numpy()
    counts = train["__target__"].value_counts()
    if len(counts) != 2:
        raise ValueError("Expected a binary target with exactly two classes.")
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    minority = train[train["__target__"] == minority_class]
    majority = train[train["__target__"] == majority_class]
    minority_sampled = resample(minority, replace=True, n_samples=len(majority), random_state=random_state)
    balanced = pd.concat([majority, minority_sampled], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced.drop(columns="__target__"), balanced["__target__"]

def train_oversampled(frame: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> tuple[Pipeline, OversamplingResult]:
    """Train the ANN after random over-sampling and evaluate on untouched test data."""
    data = normalize_telco_frame(frame)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_balanced, y_balanced = oversample_training_data(X_train, y_train, random_state)
    model = build_baseline_model(X_balanced)
    model.fit(X_balanced, y_balanced)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    result = OversamplingResult(
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1=f1_score(y_test, predictions, zero_division=0),
        roc_auc=roc_auc_score(y_test, probabilities),
        train_rows_before=len(X_train),
        train_rows_after=len(X_balanced),
    )
    return model, result
