"""Clustering task: K selection by silhouette score and cluster profiles."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import BINARY_FEATURES, NUMERIC_FEATURES, SEED

CLUSTER_INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES


def run_clustering(df: pd.DataFrame, seed: int = SEED, k_range=(2, 3, 4, 5, 6)):
    """Scale input features (never targets), test K=2..6, pick K by silhouette.

    Returns the report dict and the fitted KMeans model plus scaler.
    Only farm condition features are used; targets are excluded to avoid leakage.
    """
    X = df[CLUSTER_INPUT_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = float(silhouette_score(X_scaled, labels))
        scores.append({"k": k, "silhouette": round(score, 4)})

    best = max(scores, key=lambda s: s["silhouette"])
    selected_k = best["k"]

    kmeans = KMeans(n_clusters=selected_k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    profiles = []
    df_profile = df[CLUSTER_INPUT_FEATURES].copy()
    df_profile["cluster"] = labels
    for cluster_id in range(selected_k):
        group = df_profile[df_profile["cluster"] == cluster_id]
        center = {col: round(float(group[col].median()), 4) for col in CLUSTER_INPUT_FEATURES}
        profiles.append(
            {
                "cluster": cluster_id,
                "size": int(len(group)),
                "median_features": center,
                "plain_language_profile": _describe_cluster(group),
            }
        )

    report = {
        "task": "clustering",
        "features_used": CLUSTER_INPUT_FEATURES,
        "targets_excluded": ["yield_tons", "high_risk"],
        "k_scores": scores,
        "selected_k": selected_k,
        "selection_reason": (
            f"K={selected_k} maximised the silhouette score "
            f"({best['silhouette']}) across K=2..6 on scaled farm-condition features."
        ),
        "cluster_centers_scaled": {
            str(i): [round(float(v), 4) for v in center]
            for i, center in enumerate(kmeans.cluster_centers_)
        },
        "profiles": profiles,
    }
    return report, kmeans, scaler


def _describe_cluster(group: pd.DataFrame) -> str:
    """Write a plain-language profile from cluster medians."""
    med = group[CLUSTER_INPUT_FEATURES].median()
    parts = [f"A cluster of {len(group)} farms."]
    if med["rainfall_mm"] >= 1050:
        parts.append("Rainfall is generally high.")
    else:
        parts.append("Rainfall is generally low to moderate.")
    if med["soil_moisture_pct"] >= 50:
        parts.append("Soil moisture tends to be abundant.")
    else:
        parts.append("Soil moisture tends to be limited.")
    if med["fertilizer_kg"] >= 125:
        parts.append("Fertilizer use is relatively intensive.")
    else:
        parts.append("Fertilizer use is relatively light.")
    if med["field_size_ha"] >= 2.0:
        parts.append("Fields are on the larger side.")
    else:
        parts.append("Fields are on the smaller side.")
    if med["cooperative_support"] >= 0.5:
        parts.append("Most farms receive cooperative support.")
    else:
        parts.append("Most farms do not receive cooperative support.")
    return " ".join(parts)


def assign_clusters(kmeans, scaler, record_df: pd.DataFrame) -> int:
    """Assign a cluster label to a single farm record."""
    X = record_df[CLUSTER_INPUT_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        if pd.isna(X[col].iloc[0]):
            X[col] = 0.0
    return int(kmeans.predict(scaler.transform(X))[0])
