"""Artifact helpers: fingerprint file, JSON metrics, plots, model card metadata."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CLASSIFICATION_MODEL_PATH,
    CLUSTERING_MODEL_PATH,
    FINGERPRINT_PATH,
    MODEL_METADATA_PATH,
    PREPROCESSOR_PATH,
    REGRESSION_MODEL_PATH,
    VERIFICATION_CODE,
)


def write_fingerprint(fingerprint: str, path: Path = FINGERPRINT_PATH) -> None:
    """Persist the SHA-256 dataset fingerprint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Verification code: {VERIFICATION_CODE}\nSHA-256: {fingerprint}\n",
        encoding="utf-8",
    )


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_models(preprocessor, reg_model, clf_model, kmeans, scaler, metadata: dict) -> None:
    """Persist preprocessing, all final models and metadata for inference."""
    from sklearn.pipeline import Pipeline

    MODELS_DIR = PREPROCESSOR_PATH.parent
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(reg_model, REGRESSION_MODEL_PATH)
    joblib.dump(clf_model, CLASSIFICATION_MODEL_PATH)
    clustering_pipe = Pipeline(steps=[("scaler", scaler), ("kmeans", kmeans)])
    joblib.dump(clustering_pipe, CLUSTERING_MODEL_PATH)

    metadata = dict(metadata)
    metadata["verification_code"] = VERIFICATION_CODE
    save_json(metadata, MODEL_METADATA_PATH)


def load_models() -> dict:
    """Load all saved artifacts needed for inference."""
    return {
        "preprocessor": joblib.load(PREPROCESSOR_PATH),
        "regression": joblib.load(REGRESSION_MODEL_PATH),
        "classification": joblib.load(CLASSIFICATION_MODEL_PATH),
        "clustering": joblib.load(CLUSTERING_MODEL_PATH),
        "metadata": json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")),
    }


def plot_residuals(y_test, y_pred, path: Path) -> None:
    """Residuals-vs-predicted scatter with a zero reference line."""
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted yield_tons")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Regression residuals ({VERIFICATION_CODE})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, path: Path) -> None:
    """Draw a confusion matrix heatmap without seaborn."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["low risk (0)", "high risk (1)"])
    ax.set_yticks([0, 1], labels=["low risk (0)", "high risk (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix ({VERIFICATION_CODE})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_clusters(scaled_features, labels, path: Path) -> None:
    """2D PCA projection of the clusters."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=7483)
    coords = pca.fit_transform(scaled_features)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                         alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(f"Farm clusters ({VERIFICATION_CODE})")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_test_predictions(
    row_ids, y_reg_true, y_reg_pred, y_clf_true, y_clf_pred, y_clf_proba, clusters, path: Path
) -> pd.DataFrame:
    """Write the required test_predictions.csv evidence table."""
    df = pd.DataFrame(
        {
            "row_id": row_ids,
            "actual_yield_tons": np.round(np.asarray(y_reg_true, dtype=float), 4),
            "predicted_yield_tons": np.round(np.asarray(y_reg_pred, dtype=float), 4),
            "yield_error": np.round(
                np.asarray(y_reg_true, dtype=float) - np.asarray(y_reg_pred, dtype=float), 4
            ),
            "actual_high_risk": np.asarray(y_clf_true),
            "predicted_high_risk": np.asarray(y_clf_pred),
            "high_risk_probability": np.round(np.asarray(y_clf_proba, dtype=float), 4),
            "cluster": np.asarray(clusters),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    return df
