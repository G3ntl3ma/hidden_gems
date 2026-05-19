from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _extract_model_bundle(payload: Any) -> tuple[Any, float]:
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        threshold = float(payload.get("threshold", 0.5))
        return model, threshold
    return payload, 0.5


def _align_features(df: pd.DataFrame, payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return df
    required = payload.get("feature_columns")
    if not required:
        return df
    required_cols = [str(c) for c in required]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV missing required features: {missing}")
    extra = [c for c in df.columns if c not in required_cols]
    if extra:
        # Keep behavior strict by default; training feature space is explicit.
        raise ValueError(f"Prediction CSV contains unexpected columns: {extra}")
    return df[required_cols].copy()


def _positive_scores(model: Any, df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(df), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError("Loaded model does not support probability-style predictions.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/models/model.joblib", help="Model path")
    ap.add_argument("--csv", required=True, help="CSV with feature columns")
    ap.add_argument("--out", default="artifacts/models/predictions.json", help="Output predictions")
    ap.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    args = ap.parse_args()

    payload = joblib.load(args.model)
    model, default_threshold = _extract_model_bundle(payload)
    threshold = default_threshold if args.threshold is None else float(args.threshold)

    df = pd.read_csv(args.csv)
    df = _align_features(df, payload)
    scores = _positive_scores(model, df)
    preds = (scores >= threshold).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "threshold": threshold,
        "predictions": [int(p) for p in preds],
        "probabilities": [float(s) for s in scores],
    }
    out_path.write_text(json.dumps(out) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

