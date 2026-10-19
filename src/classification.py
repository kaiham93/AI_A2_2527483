"""Classification task: majority baseline versus candidate models."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Always predict the most frequent training class."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        vals, counts = np.unique(y, return_counts=True)
        self.majority_ = vals[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        idx = list(self.classes_).index(self.majority_)
        proba[:, idx] = 1.0
        return proba


def build_candidates(preprocessor) -> dict[str, Pipeline]:
    """Return classification candidates: majority baseline, logistic regression, random forest."""
    return {
        "majority_baseline": Pipeline(
            steps=[("preprocess", preprocessor), ("model", MajorityBaseline())]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=7483),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200, random_state=7483, n_jobs=-1
                    ),
                ),
            ]
        ),
    }


def evaluate(name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute precision, recall, F1, confusion matrix and ROC-AUC."""
    preds = pipe.predict(X_test)
    metrics = {
        "model": name,
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def run_classification(X_train, X_test, y_train, y_test, preprocessor):
    """Train all candidates, select the best by test F1, return metrics and model."""
    candidates = build_candidates(preprocessor)
    results = []
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(name, pipe, X_test, y_test))

    best = max(results, key=lambda r: r["f1"])
    reason = (
        f"Selected {best['model']} because it achieved the highest test F1 "
        f"({best['f1']}) among {len(results)} candidates, with precision "
        f"{best['precision']}, recall {best['recall']} and ROC-AUC {best['roc_auc']}."
    )
    report = {
        "task": "classification",
        "target": "high_risk",
        "positive_class": 1,
        "baseline": results[0],
        "candidates": results,
        "selected_model": best["model"],
        "selection_reason": reason,
        "test_rows": int(len(y_test)),
    }
    return report, fitted[best["model"]]
