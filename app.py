"""FastAPI service exposing validated inference for the harvest intelligence system.

Endpoints:
    GET  /health      - liveness probe
    GET  /model-info  - model metadata and verification code
    POST /predict     - validated single-record prediction
"""
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.artifacts import load_models
from src.config import VERIFICATION_CODE
from predict import predict_one

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=(
        "Classroom decision-support prototype: harvest-yield estimation, "
        "high-production-risk classification and farm clustering for a "
        "Musanze potato cooperative. Synthetic data only. "
        f"Verification code: {VERIFICATION_CODE}."
    ),
    version="1.0.0",
)

_BUNDLE = None


def get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        try:
            _BUNDLE = load_models()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model artifacts missing: {exc}. Run 'python run_all.py' first.",
            )
    return _BUNDLE


class FarmRecord(BaseModel):
    rainfall_mm: float = Field(..., ge=600, le=1400, description="Annual rainfall in mm")
    avg_temp_c: float = Field(..., ge=12, le=26, description="Average temperature in Celsius")
    soil_moisture_pct: float = Field(..., ge=20, le=80, description="Soil moisture percentage")
    fertilizer_kg: float = Field(..., ge=0, le=250, description="Fertilizer applied in kg")
    field_size_ha: float = Field(..., ge=0.2, le=4.0, description="Field size in hectares")
    season: Literal["A", "B"] = "A"
    cooperative_support: Literal[0, 1] = 0


class PredictionResponse(BaseModel):
    verification_code: str
    yield_tons: float
    high_risk: int
    high_risk_probability: float
    cluster: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "verification_code": VERIFICATION_CODE}


@app.get("/model-info")
def model_info() -> dict:
    bundle = get_bundle()
    meta = bundle["metadata"]
    return {
        "verification_code": VERIFICATION_CODE,
        "student": meta.get("student"),
        "registration": meta.get("registration"),
        "seed": meta.get("seed"),
        "regression_model": meta.get("regression_model"),
        "classification_model": meta.get("classification_model"),
        "clustering_k": meta.get("clustering_k"),
        "features": meta.get("features"),
        "dataset_fingerprint": meta.get("dataset_fingerprint"),
        "intended_use": (
            "Classroom decision-support prototype for a Musanze potato cooperative. "
            "Synthetic data. Must not replace an agronomist or determine access to "
            "finance, insurance or public services."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord) -> PredictionResponse:
    bundle = get_bundle()
    result = predict_one(record.model_dump(), bundle)
    return PredictionResponse(**result)
