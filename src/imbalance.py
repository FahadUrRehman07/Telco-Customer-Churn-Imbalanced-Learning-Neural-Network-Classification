"""Class-imbalance diagnostics for the Telco churn dataset.

This module keeps the analysis separate from model training so the project can
report the target distribution before applying any resampling strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .preprocessing import normalize_telco_frame, split_features_target


@dataclass(frozen=True)
class ImbalanceReport:
    """Summary of the binary target distribution."""

    total_rows: int
    positive_count: int
    negative_count: int
    positive_rate: float
    negative_rate: float
    minority_to_majority_ratio: float


def analyze_target_distribution(frame: pd.DataFrame) -> ImbalanceReport:
    """Return class counts and rates for the normalized churn target."""
    data = normalize_telco_frame(frame)
    _, target = split_features_target(data)

    positive_count = int(target.sum())
    total_rows = int(target.shape[0])
    negative_count = total_rows - positive_count
    majority_count = max(positive_count, negative_count)
    minority_count = min(positive_count, negative_count)

    return ImbalanceReport(
        total_rows=total_rows,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_rate=positive_count / total_rows,
        negative_rate=negative_count / total_rows,
        minority_to_majority_ratio=minority_count / majority_count,
    )


def majority_class_accuracy(frame: pd.DataFrame) -> float:
    """Return the accuracy of always predicting the majority class."""
    report = analyze_target_distribution(frame)
    return max(report.positive_rate, report.negative_rate)
