"""Leakage-safe preprocessing utilities for the Telco churn project.

The key rule in this module is that learned preprocessing state is fitted on
training data only. The resulting sklearn Pipeline can then be used unchanged
for validation, test, and future inference data.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"


def normalize_telco_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco dataframe.

    The IBM Telco dataset commonly stores ``TotalCharges`` as text because of
    blank entries. Those blanks are converted to missing values and the column
    is converted to numeric so imputation can happen inside the pipeline.
    """

    data = frame.copy()
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    if TARGET_COLUMN in data.columns:
        data[TARGET_COLUMN] = data[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype("Int64")

    return data


def split_features_target(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a normalized dataframe into features and target."""

    if target_column not in frame.columns:
        raise KeyError(f"Target column {target_column!r} was not found")

    X = frame.drop(columns=[target_column])
    y = frame[target_column].astype(int)
    return X, y


def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Build a preprocessing transformer from feature names only.

    No values are learned here. Imputation statistics, scaling parameters, and
    category vocabularies are learned later when ``fit`` is called on training
    data only.
    """

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessing_pipeline(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Return a ready-to-fit leakage-safe preprocessing pipeline."""

    if X_train.empty:
        raise ValueError("X_train must contain at least one row and one column")
    return build_preprocessor(X_train)


def transformed_feature_names(
    fitted_preprocessor: ColumnTransformer,
) -> Iterable[str]:
    """Return output feature names after fitting the preprocessor."""

    return fitted_preprocessor.get_feature_names_out()
