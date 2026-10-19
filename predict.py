"""Command-line prediction from saved artifacts.

Usage:
    python predict.py --json samples/valid.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.artifacts import load_models
from src.config import FEATURE_COLUMNS, VERIFICATION_CODE

REQUIRED_FIELDS = FEATURE_COLUMNS


def validate_record(record: dict) -> list[str]:
    """Return a list of validation errors for one input record (empty if valid)."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing field: {field}")
    if errors:
        return errors

    numeric_ranges = {
        "rainfall_mm": (600.0, 1400.0),
        "avg_temp_c": (12.0, 26.0),
        "soil_moisture_pct": (20.0, 80.0),
        "fertilizer_kg": (0.0, 250.0),
        "field_size_ha": (0.2, 4.0),
    }
    for field, (lo, hi) in numeric_ranges.items():
        value = record[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{field} must be a number")
        elif not (lo <= float(value) <= hi):
            errors.append(f"{field} must be between {lo} and {hi}")

    if record.get("season") not in ("A", "B"):
        errors.append("season must be 'A' or 'B'")
    if record.get("cooperative_support") not in (0, 1):
        errors.append("cooperative_support must be 0 or 1")
    return errors


def predict_one(record: dict, bundle: dict) -> dict:
    """Run regression, classification and clustering for one record."""
    df = pd.DataFrame([record], columns=REQUIRED_FIELDS)
    yield_estimate = float(bundle["regression"].predict(df)[0])
    risk_label = int(bundle["classification"].predict(df)[0])
    risk_probability = float(bundle["classification"].predict_proba(df)[0][1])
    from src.clustering import CLUSTER_INPUT_FEATURES

    cluster = int(bundle["clustering"].predict(df[CLUSTER_INPUT_FEATURES])[0])
    return {
        "verification_code": VERIFICATION_CODE,
        "yield_tons": round(yield_estimate, 4),
        "high_risk": risk_label,
        "high_risk_probability": round(risk_probability, 4),
        "cluster": cluster,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one farm record")
    parser.add_argument("--json", required=True, help="Path to a JSON file with one record")
    args = parser.parse_args()

    path = Path(args.json)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    record = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_record(record)
    if errors:
        print("Invalid input record:")
        for e in errors:
            print(f"  - {e}")
        return 1

    bundle = load_models()
    result = predict_one(record, bundle)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
