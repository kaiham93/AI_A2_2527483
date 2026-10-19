"""Reproducible entry point: recreate data, clean, train all tasks, write artifacts.

Usage:
    python run_all.py --student-id ug2527483@ines.ac.rw --seed 7483
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.generate_data import generate_dataframe, save_csv, CSV_NAME
from src import artifacts
from src.classification import run_classification
from src.clustering import CLUSTER_INPUT_FEATURES, run_clustering
from src.config import (
    ARTIFACTS_DIR,
    CLASSIFICATION_METRICS_PATH,
    CLUSTERING_REPORT_PATH,
    CSV_PATH,
    INSTITUTIONAL_USERNAME,
    PLOTS_DIR,
    REGISTRATION,
    REGRESSION_METRICS_PATH,
    SEED,
    STUDENT_NAME,
    TEST_PREDICTIONS_PATH,
    VERIFICATION_CODE,
)
from src.preprocessing import build_preprocessor, clean_dataset, split_data
from src.regression import run_regression
from src.validation import ValidationError, compute_fingerprint, validate_dataset, validate_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service pipeline")
    parser.add_argument("--student-id", required=True, help="Institutional username, e.g. ug2527483@ines.ac.rw")
    parser.add_argument("--seed", required=True, type=int, help="Personal seed LLLL (7483)")
    args = parser.parse_args()

    print("=" * 70)
    print("Musanze Harvest Intelligence Service - full pipeline")
    print(f"Student: {STUDENT_NAME}")
    print(f"Registration: {REGISTRATION}")
    print(f"Verification code: {VERIFICATION_CODE}")
    print("=" * 70)

    try:
        validate_seed(args.seed)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.student_id.lower() != INSTITUTIONAL_USERNAME.lower():
        print(
            f"WARNING: student-id {args.student_id} does not match expected "
            f"{INSTITUTIONAL_USERNAME}; continuing with seed validation only.",
            file=sys.stderr,
        )

    print("\n[1/7] Generating dataset ...")
    raw = generate_dataframe(seed=args.seed)
    save_csv(raw, CSV_PATH)
    fingerprint = compute_fingerprint(CSV_PATH.read_bytes())
    artifacts.write_fingerprint(fingerprint)
    print(f"  rows (raw): {len(raw)}")
    print(f"  SHA-256 fingerprint: {fingerprint}")

    print("\n[2/7] Validating dataset ...")
    issues = validate_dataset(raw)
    for issue in issues:
        print(f"  detected: {issue}")

    print("\n[3/7] Cleaning and splitting ...")
    cleaned, clean_stats = clean_dataset(raw)
    print(f"  {clean_stats}")
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(cleaned)
    print(f"  train rows: {len(X_train)}, test rows: {len(X_test)} (random_state={args.seed})")

    preprocessor = build_preprocessor()

    print("\n[4/7] Regression ...")
    reg_report, reg_model = run_regression(
        X_train, X_test, y_reg_train, y_reg_test, build_preprocessor()
    )
    artifacts.save_json(reg_report, REGRESSION_METRICS_PATH)
    for row in reg_report["candidates"]:
        print(f"  {row['model']}: MAE={row['mae']} RMSE={row['rmse']} R2={row['r2']}")
    print(f"  selected: {reg_report['selected_model']}")

    print("\n[5/7] Classification ...")
    clf_report, clf_model = run_classification(
        X_train, X_test, y_clf_train, y_clf_test, build_preprocessor()
    )
    artifacts.save_json(clf_report, CLASSIFICATION_METRICS_PATH)
    for row in clf_report["candidates"]:
        print(
            f"  {row['model']}: P={row['precision']} R={row['recall']} "
            f"F1={row['f1']} AUC={row['roc_auc']}"
        )
    print(f"  selected: {clf_report['selected_model']}")

    print("\n[6/7] Clustering ...")
    cluster_report, kmeans, scaler = run_clustering(cleaned)
    artifacts.save_json(cluster_report, CLUSTERING_REPORT_PATH)
    for s in cluster_report["k_scores"]:
        print(f"  K={s['k']}: silhouette={s['silhouette']}")
    print(f"  selected K: {cluster_report['selected_k']}")
    for prof in cluster_report["profiles"]:
        print(f"  cluster {prof['cluster']} (n={prof['size']}): {prof['plain_language_profile']}")

    print("\n[7/7] Test predictions, plots and saved models ...")
    y_reg_pred = reg_model.predict(X_test)
    y_clf_pred = clf_model.predict(X_test)
    try:
        y_clf_proba = clf_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_clf_proba = np.asarray(y_clf_pred, dtype=float)
    clusters = kmeans.predict(scaler.transform(_cluster_features(X_test)))
    if "row_id" in X_test.columns:
        row_ids = X_test["row_id"].values
    else:
        row_ids = X_test.index.values
    artifacts.build_test_predictions(
        row_ids, y_reg_test, y_reg_pred, y_clf_test, y_clf_pred, y_clf_proba,
        clusters, TEST_PREDICTIONS_PATH,
    )
    artifacts.plot_residuals(y_reg_test, y_reg_pred, PLOTS_DIR / "residuals.png")
    artifacts.plot_confusion_matrix(clf_report["candidates"][-1]["confusion_matrix"], PLOTS_DIR / "confusion_matrix.png")
    X_cluster = cleaned[CLUSTER_INPUT_FEATURES].copy()
    for col in ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha"]:
        X_cluster[col] = X_cluster[col].fillna(X_cluster[col].median())
    artifacts.plot_clusters(scaler.transform(X_cluster), kmeans.labels_, PLOTS_DIR / "clusters.png")

    metadata = {
        "student": STUDENT_NAME,
        "registration": REGISTRATION,
        "seed": args.seed,
        "verification_code": VERIFICATION_CODE,
        "regression_model": reg_report["selected_model"],
        "classification_model": clf_report["selected_model"],
        "clustering_k": cluster_report["selected_k"],
        "features": ["rainfall_mm", "avg_temp_c", "soil_moisture_pct",
                     "fertilizer_kg", "field_size_ha", "season", "cooperative_support"],
        "dataset_fingerprint": fingerprint,
        "rows_raw": len(raw),
        "rows_cleaned": len(cleaned),
    }
    artifacts.save_models(preprocessor, reg_model, clf_model, kmeans, scaler, metadata)

    print("\nArtifacts written:")
    for p in sorted(ARTIFACTS_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ARTIFACTS_DIR.parent)}")
    print(f"\nDone. Verification code: {VERIFICATION_CODE}")
    return 0


def _cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.clustering import CLUSTER_INPUT_FEATURES
    out = df[CLUSTER_INPUT_FEATURES].copy()
    for col in ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


if __name__ == "__main__":
    raise SystemExit(main())
