"""Cleaning and preprocessing with train-only fitting (no target leakage)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    SEED,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows and return cleaning statistics."""
    n_before = len(df)
    cleaned = df.drop_duplicates(subset=FEATURE_COLUMNS + [TARGET_REGRESSION]).reset_index(
        drop=True
    )
    if "row_id" in cleaned.columns:
        cleaned["row_id"] = np.arange(1, len(cleaned) + 1)
    stats = {
        "rows_before": n_before,
        "rows_after": len(cleaned),
        "duplicates_removed": n_before - len(cleaned),
    }
    return cleaned, stats


def build_preprocessor() -> ColumnTransformer:
    """Build the shared ColumnTransformer used by all three tasks.

    Numeric features: median imputation + standard scaling.
    Binary features: median imputation only (already 0/1).
    Categorical season: most-frequent imputation + one-hot encoding.
    Everything is fit on training data only via the model pipelines.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame, seed: int = SEED):
    """Fixed train/test split with random_state=seed.

    Returns X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test.
    """
    X = df[FEATURE_COLUMNS]
    y_reg = df[TARGET_REGRESSION]
    y_clf = df[TARGET_CLASSIFICATION]
    return train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=seed
    )
