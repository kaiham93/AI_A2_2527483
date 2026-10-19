"""Synthetic dataset generator for the Musanze Harvest Intelligence Service.

Recreates data/AI_A2_2527483.csv deterministically from the personal seed
LLLL = 7483 (last four digits of registration number 25/27483).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7483
VERIFICATION_CODE = "AI-A2-7483"
REGISTRATION = "25/27483"
STUDENT_NAME = "KHALIL ABDEEN HAMID MUSA"
CSV_NAME = "AI_A2_2527483.csv"

BASE_ROWS = 240 + (SEED % 41)  # 261 rows before duplicates
MISSING_COUNT = round(0.05 * BASE_ROWS)  # 13 missing per targeted column
DUPLICATE_COUNT = 3

FEATURE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]
TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_REGRESSION, TARGET_CLASSIFICATION]

RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}


def generate_dataframe(seed: int = SEED) -> pd.DataFrame:
    """Generate the raw synthetic farm dataset with deliberate defects.

    Uses numpy.random.default_rng(seed). Injects missing values into
    soil_moisture_pct and fertilizer_kg, then appends three exact duplicate
    rows. Returns the resulting DataFrame with a row_id column.
    """
    rng = np.random.default_rng(seed)
    n = BASE_ROWS

    rainfall = rng.uniform(600, 1400, n)
    avg_temp = rng.uniform(12, 26, n)
    soil_moisture = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    field_size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], n)
    support = rng.integers(0, 2, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(0.0, 0.80, n)

    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil_moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(avg_temp - 18.0)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": avg_temp,
            "soil_moisture_pct": soil_moisture,
            "fertilizer_kg": fertilizer,
            "field_size_ha": field_size,
            "season": season,
            "cooperative_support": support.astype(int),
            "yield_tons": yield_tons,
        }
    )

    yield_per_ha = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (yield_per_ha < yield_per_ha.median()).astype(int)

    # Deliberate data problems: missing values chosen by the same generator.
    missing_sm = rng.choice(n, size=MISSING_COUNT, replace=False)
    remaining = np.setdiff1d(np.arange(n), missing_sm)
    missing_fert = rng.choice(remaining, size=MISSING_COUNT, replace=False)
    df.loc[missing_sm, "soil_moisture_pct"] = np.nan
    df.loc[missing_fert, "fertilizer_kg"] = np.nan

    # Three exact duplicate rows.
    dup_idx = rng.choice(n, size=DUPLICATE_COUNT, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    df.insert(0, "row_id", np.arange(1, len(df) + 1))
    return df


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Save the dataset with four decimal places for numeric values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def main() -> None:
    csv_path = Path(__file__).resolve().parent / CSV_NAME
    df = generate_dataframe()
    save_csv(df, csv_path)
    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"Student: {STUDENT_NAME} ({REGISTRATION})")
    print(f"Seed: {SEED}")
    print(f"Rows written: {len(df)} (base {BASE_ROWS} + {DUPLICATE_COUNT} duplicates)")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
