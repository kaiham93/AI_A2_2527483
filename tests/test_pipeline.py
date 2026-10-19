"""Pytest suite for the Musanze Harvest Intelligence Service."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate_dataframe, save_csv
from src.config import SEED, VERIFICATION_CODE
from src.validation import ValidationError, compute_fingerprint, validate_seed
from predict import validate_record, predict_one

VALID_RECORD = {
    "rainfall_mm": 1000.0,
    "avg_temp_c": 18.0,
    "soil_moisture_pct": 45.0,
    "fertilizer_kg": 120.0,
    "field_size_ha": 1.5,
    "season": "A",
    "cooperative_support": 1,
}


def test_seed_validation_rejects_wrong_seed():
    with pytest.raises(ValidationError):
        validate_seed(1234)


def test_seed_validation_accepts_personal_seed():
    validate_seed(SEED)


def test_dataset_reproducible(tmp_path):
    df1 = generate_dataframe(seed=SEED)
    df2 = generate_dataframe(seed=SEED)
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    save_csv(df1, p1)
    save_csv(df2, p2)
    assert compute_fingerprint(p1.read_bytes()) == compute_fingerprint(p2.read_bytes())
    assert len(df1) == 240 + (SEED % 41) + 3


def test_valid_inference():
    bundle = _load_bundle()
    result = predict_one(VALID_RECORD, bundle)
    assert result["verification_code"] == VERIFICATION_CODE
    assert isinstance(result["yield_tons"], float)
    assert result["high_risk"] in (0, 1)
    assert 0.0 <= result["high_risk_probability"] <= 1.0
    assert isinstance(result["cluster"], int)


def test_missing_field_rejected():
    bad = dict(VALID_RECORD)
    del bad["rainfall_mm"]
    errors = validate_record(bad)
    assert any("missing field" in e for e in errors)


def test_wrong_type_rejected():
    bad = dict(VALID_RECORD)
    bad["fertilizer_kg"] = "lots"
    errors = validate_record(bad)
    assert any("must be a number" in e for e in errors)


def test_range_error_rejected():
    bad = dict(VALID_RECORD)
    bad["rainfall_mm"] = 50.0
    errors = validate_record(bad)
    assert any("between" in e for e in errors)


def test_invalid_season_rejected():
    bad = dict(VALID_RECORD)
    bad["season"] = "C"
    errors = validate_record(bad)
    assert any("season" in e for e in errors)


def test_api_health_and_predict():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["verification_code"] == VERIFICATION_CODE

    r = client.post("/predict", json=VALID_RECORD)
    assert r.status_code == 200
    body = r.json()
    assert body["verification_code"] == VERIFICATION_CODE
    assert "yield_tons" in body and "cluster" in body

    r = client.post("/predict", json={"rainfall_mm": 100})
    assert 400 <= r.status_code < 500


def _load_bundle():
    from src.artifacts import load_models

    return load_models()
