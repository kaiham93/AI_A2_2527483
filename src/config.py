"""Shared configuration for the Musanze Harvest Intelligence Service."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED = 7483
VERIFICATION_CODE = "AI-A2-7483"
REGISTRATION = "25/27483"
STUDENT_NAME = "KHALIL ABDEEN HAMID MUSA"
INSTITUTIONAL_USERNAME = "ug2527483@ines.ac.rw"

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SAMPLES_DIR = PROJECT_ROOT / "samples"

CSV_PATH = DATA_DIR / "AI_A2_2527483.csv"
FINGERPRINT_PATH = ARTIFACTS_DIR / "data_fingerprint.txt"

FEATURE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]
NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"

RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
ALLOWED_SEASONS = ["A", "B"]

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.joblib"
CLUSTERING_MODEL_PATH = MODELS_DIR / "clustering_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REGRESSION_METRICS_PATH = ARTIFACTS_DIR / "regression_metrics.json"
CLASSIFICATION_METRICS_PATH = ARTIFACTS_DIR / "classification_metrics.json"
CLUSTERING_REPORT_PATH = ARTIFACTS_DIR / "clustering_report.json"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
