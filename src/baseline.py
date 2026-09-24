"""Baseline ANN workflow for the Telco churn project.

The baseline intentionally keeps the experiment simple: a stratified holdout,
the leakage-safe preprocessor from ``src.preprocessing``, and a small
feed-forward neural network. Later days can change only the sampling strategy
or decision threshold so comparisons remain interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessing_pipeline, normalize_telco_frame, split_features_target


@dataclass(frozen=True)
class BaselineResult:
    """Metrics produced by the baseline model on the untouched test set."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def build_baseline_model(X_train: pd.DataFrame) -> Pipeline:
    """Create the leakage-safe preprocessing + ANN baseline pipeline."""
    preprocessor = build_preprocessing_pipeline(X_train)
    classifier = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_baseline(frame: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> tuple[Pipeline, BaselineResult]:
    """Train the baseline and evaluate it once on a stratified test split."""
    data = normalize_telco_frame(frame)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = build_baseline_model(X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    result = BaselineResult(
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1=f1_score(y_test, predictions, zero_division=0),
        roc_auc=roc_auc_score(y_test, probabilities),
    )
    return model, result


def load_telco_csv(path: str) -> pd.DataFrame:
    """Load a Telco CSV without target-dependent preprocessing."""
    return pd.read_csv(path)


if __name__ == "__main__":
    raise SystemExit("Import train_baseline() from a notebook or experiment script.")