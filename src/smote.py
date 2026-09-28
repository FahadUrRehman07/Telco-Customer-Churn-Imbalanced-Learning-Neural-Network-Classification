"""SMOTE experiment for the Telco churn project."""

from dataclasses import dataclass
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .baseline import build_baseline_model
from .preprocessing import normalize_telco_frame, split_features_target

@dataclass(frozen=True)
class SmoteResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_rows_before: int
    train_rows_after: int

def train_smote(frame: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> tuple[Pipeline, SmoteResult]:
    """Train the ANN after SMOTE and evaluate on untouched test data."""
    data = normalize_telco_frame(frame)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    base = build_baseline_model(X_train)
    preprocessor = base.named_steps["preprocessor"]
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    X_balanced, y_balanced = SMOTE(random_state=random_state).fit_resample(X_train_encoded, y_train)
    classifier = base.named_steps["classifier"]
    classifier.fit(X_balanced, y_balanced)
    predictions = classifier.predict(X_test_encoded)
    probabilities = classifier.predict_proba(X_test_encoded)[:, 1]
    result = SmoteResult(accuracy_score(y_test, predictions), precision_score(y_test, predictions, zero_division=0), recall_score(y_test, predictions, zero_division=0), f1_score(y_test, predictions, zero_division=0), roc_auc_score(y_test, probabilities), len(X_train), len(X_balanced))
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)]), result
