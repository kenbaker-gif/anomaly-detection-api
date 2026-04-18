"""
model.py — Model loader and inference engine.
Loaded once at startup and reused across requests.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.meta: dict = {}
        self.feature_cols: list[str] = []
        self.threshold: float = 0.0
        self._loaded = False

    def load(self):
        model_path = MODELS_DIR / "isolation_forest.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        meta_path = MODELS_DIR / "model_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python -m app.train` first to train and save the model."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feature_cols = self.meta["feature_cols"]
        self.threshold = self.meta["threshold"]
        self._loaded = True
        print(f"[model] Loaded {self.meta['model_type']} — version {self.meta['version']}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self.meta.get("version", "unknown")

    def _extract_features(self, transaction: dict) -> np.ndarray:
        return np.array([[transaction[col] for col in self.feature_cols]])

    def _build_explanation(self, raw_features: np.ndarray, scaled_features: np.ndarray) -> dict:
        """
        Explain the anomaly by finding which features deviate most from the
        training distribution (in standard-deviation units after scaling).
        """
        deviations = np.abs(scaled_features[0])  # already z-scored by scaler
        top_indices = np.argsort(deviations)[::-1][:5]

        top_features = [self.feature_cols[i] for i in top_indices]
        feature_contributions = {
            self.feature_cols[i]: round(float(scaled_features[0][i]), 4)
            for i in top_indices
        }
        return {
            "top_features": top_features,
            "feature_contributions": feature_contributions,
        }

    def _normalize_score(self, score: float) -> float:
        """
        Map Isolation Forest decision_function score to [0, 1] confidence.
        Scores are typically in [-0.5, 0.5]. We flip so that higher = more fraud.
        """
        # Clip to a reasonable range and normalize
        clipped = np.clip(score, -0.5, 0.5)
        normalized = (0.5 - clipped)  # flip: low IF score → high fraud confidence
        return round(float(np.clip(normalized, 0.0, 1.0)), 4)

    def predict(self, transaction: dict) -> dict:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call detector.load() first.")

        raw = self._extract_features(transaction)
        scaled = self.scaler.transform(raw)

        score = float(self.model.decision_function(scaled)[0])
        is_fraud = score < self.threshold
        confidence = self._normalize_score(score)
        explanation = self._build_explanation(raw, scaled)

        return {
            "is_fraud": is_fraud,
            "label": "FRAUD" if is_fraud else "NORMAL",
            "anomaly_score": round(score, 6),
            "confidence": confidence,
            "explanation": explanation,
            "model_version": self.version,
        }

    def predict_batch(self, transactions: list[dict]) -> list[dict]:
        if not self._loaded:
            raise RuntimeError("Model not loaded.")
        if not transactions:
            return []

        raw = np.array([[t[col] for col in self.feature_cols] for t in transactions])
        scaled = self.scaler.transform(raw)
        scores = self.model.decision_function(scaled)

        results = []
        for i, (transaction, score) in enumerate(zip(transactions, scores)):
            is_fraud = float(score) < self.threshold
            confidence = self._normalize_score(float(score))
            explanation = self._build_explanation(raw[i : i + 1], scaled[i : i + 1])
            results.append({
                "is_fraud": is_fraud,
                "label": "FRAUD" if is_fraud else "NORMAL",
                "anomaly_score": round(float(score), 6),
                "confidence": confidence,
                "explanation": explanation,
                "model_version": self.version,
            })
        return results


# Singleton — imported and shared across the app
detector = AnomalyDetector()