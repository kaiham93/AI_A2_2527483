"""Regression task: mean baseline versus candidate models."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predict the training mean of the target for every row."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def build_candidates(preprocessor) -> dict[str, Pipeline]:
    """Return regression candidates: mean baseline, linear regression, gradient boosting."""
    return {
        "mean_baseline": Pipeline(
            steps=[("preprocess", preprocessor), ("model", MeanBaseline())]
        ),
        "linear_regression": Pipeline(
            steps=[("preprocess", preprocessor), ("model", LinearRegression())]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                (
                    "preprocess", preprocessor
                ),
                (
                    "model",
                    GradientBoostingRegressor(random_state=7483),
                ),
            ]
        ),
    }


def evaluate(name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute MAE, RMSE and R² on the held-out test set."""
    preds = pipe.predict(X_test)
    return {
        "model": name,
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
    }


def run_regression(X_train, X_test, y_train, y_test, preprocessor):
    """Train all candidates, select the best by test MAE, return metrics and model."""
    candidates = build_candidates(preprocessor)
    results = []
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(name, pipe, X_test, y_test))

    selected = min(results, key=lambda r: r["mae"])
    reason = (
        f"Selected {selected['model']} because it achieved the lowest test MAE "
        f"({selected['mae']}) among {len(results)} candidates, with RMSE "
        f"{selected['rmse']} and R2 {selected['r2']}."
    )
    report = {
        "task": "regression",
        "target": "yield_tons",
        "baseline": results[0],
        "candidates": results,
        "selected_model": selected["model"],
        "selection_reason": reason,
        "test_rows": int(len(y_test)),
    }
    return report, fitted[selected["model"]]
