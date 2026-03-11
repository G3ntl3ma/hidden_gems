from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/artifacts/model.joblib", help="Model path")
    ap.add_argument("--csv", required=True, help="CSV with feature columns")
    ap.add_argument("--out", default="models/artifacts/predictions.json", help="Output predictions")
    args = ap.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.csv)
    preds = model.predict(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"predictions": [p.item() if hasattr(p, "item") else p for p in preds]}) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

