"""Dataset validation: schema, ranges, missing values, duplicates, leakage."""
import hashlib

import numpy as np
import pandas as pd

from src.config import (
    ALLOWED_SEASONS,
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANGES,
    SEED,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_REGRESSION, TARGET_CLASSIFICATION]
LEAKY_COLUMNS = ["yield_per_ha", "yield_tons_per_ha"]


class ValidationError(Exception):
    """Raised when the dataset or configuration fails validation."""


def validate_seed(seed: int) -> None:
    """Reject any seed that is not the personal seed LLLL."""
    if seed != SEED:
        raise ValidationError(
            f"Invalid seed {seed}: this submission requires seed {SEED} (LLLL of {27483})."
        )


def compute_fingerprint(csv_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of the raw CSV bytes."""
    return hashlib.sha256(csv_bytes).hexdigest()


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Validate schema and content of the raw dataset.

    Returns a list of human-readable warning strings (missing counts,
    duplicates found). Raises ValidationError on structural problems.
    """
    problems = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")

    unexpected = [c for c in df.columns if c not in REQUIRED_COLUMNS + ["row_id"]]
    if unexpected:
        raise ValidationError(f"Unexpected columns (possible leakage): {unexpected}")

    for col in LEAKY_COLUMNS:
        if col in df.columns:
            raise ValidationError(f"Target leakage column detected: {col}")

    for col in NUMERIC_FEATURES + [TARGET_REGRESSION]:
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValidationError(f"Column {col} must be numeric, got {df[col].dtype}.")

    for col, (lo, hi) in RANGES.items():
        observed = df[col].dropna()
        if observed.empty or observed.min() < lo or observed.max() > hi:
            raise ValidationError(
                f"Column {col} has values outside expected range [{lo}, {hi}]."
            )

    bad_season = set(df["season"].dropna().unique()) - set(ALLOWED_SEASONS)
    if bad_season:
        raise ValidationError(f"Invalid season values: {bad_season}")

    for col in BINARY_FEATURES:
        vals = set(df[col].dropna().unique())
        if not vals.issubset({0, 1}):
            raise ValidationError(f"Column {col} must be binary 0/1, got {vals}")

    if df[TARGET_REGRESSION].isna().any():
        raise ValidationError("Target yield_tons contains missing values.")
    if df[TARGET_REGRESSION].min() < 0.50:
        raise ValidationError("Target yield_tons below the 0.50 clip minimum.")

    for col in ["soil_moisture_pct", "fertilizer_kg"]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            problems.append(f"{col}: {n_missing} missing values detected")

    n_dupes = int(df.duplicated(subset=REQUIRED_COLUMNS).sum())
    if n_dupes:
        problems.append(f"{n_dupes} exact duplicate rows detected")

    return problems
