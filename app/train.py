"""
train.py — Train the Isolation Forest anomaly detection model.

Usage:
    python -m app.train                        # uses data/creditcard.csv
    python -m app.train --data path/to/file.csv
    python -m app.train --contamination 0.002
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_COLS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23",
    "V24", "V25", "V26", "V27", "V28", "Amount",
]


def load_data(csv_path: str) -> pd.DataFrame:
    print(f"[train] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[train] Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,}  |  Normal: {(df['Class'] == 0).sum():,}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def train(csv_path: str = "data/creditcard.csv", contamination: float = 0.002):
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data(csv_path)

    # Use only normal (non-fraud) transactions for unsupervised training —
    # this is the standard approach: teach the model what "normal" looks like.
    normal_df = df[df["Class"] == 0][FEATURE_COLS]
    X_train = normal_df.values

    # Scale features — especially important for Time and Amount
    print("[train] Fitting StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Compute per-feature means and stds (used later for explanation)
    feature_means = scaler.mean_.tolist()
    feature_stds = scaler.scale_.tolist()

    # Train Isolation Forest
    print(f"[train] Training Isolation Forest  (contamination={contamination}) ...")
    t0 = time.time()
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    model.fit(X_scaled)
    elapsed = time.time() - t0
    print(f"[train] Training done in {elapsed:.1f}s")

    # Compute decision threshold on training data (99th percentile of scores)
    scores = model.decision_function(X_scaled)
    threshold = float(np.percentile(scores, 1))  # bottom 1% = anomalies
    print(f"[train] Anomaly threshold (1st pct): {threshold:.6f}")

    # Sanity-check on full dataset
    X_all = df[FEATURE_COLS].values
    X_all_scaled = scaler.transform(X_all)
    all_scores = model.decision_function(X_all_scaled)
    fraud_scores = all_scores[df["Class"] == 1]
    normal_scores = all_scores[df["Class"] == 0]
    print(f"[train] Fraud  score: mean={fraud_scores.mean():.4f}  min={fraud_scores.min():.4f}")
    print(f"[train] Normal score: mean={normal_scores.mean():.4f}  max={normal_scores.max():.4f}")

    # Save artifacts
    version = f"v{int(time.time())}"

    model_path = MODELS_DIR / "isolation_forest.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path = MODELS_DIR / "model_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "version": version,
        "model_type": "IsolationForest",
        "n_estimators": 200,
        "contamination": contamination,
        "feature_cols": FEATURE_COLS,
        "feature_count": len(FEATURE_COLS),
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "threshold": threshold,
        "training_samples": int(len(X_train)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[train] Saved model  → {model_path}")
    print(f"[train] Saved scaler → {scaler_path}")
    print(f"[train] Saved meta   → {meta_path}")
    print(f"[train] Model version: {version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--contamination", type=float, default=0.002)
    args = parser.parse_args()
    train(csv_path=args.data, contamination=args.contamination)