from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train(df: pd.DataFrame, *, target_col: str) -> tuple[Pipeline, dict]:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe columns")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    model = LogisticRegression(max_iter=500)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, preds)), "n_test": int(len(y_test))}
    return pipe, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training CSV")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--out", default="models/artifacts/model.joblib", help="Output model path")
    ap.add_argument(
        "--metrics-out", default="models/artifacts/metrics.json", help="Output metrics path"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    model, metrics = train(df, target_col=args.target)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Saved model to {out_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

