from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from scripts._runtime import bootstrap_project_root, local_data_file, model_artifact_file

bootstrap_project_root()

from models.clustering import build_feature_matrix
from models.hidden_gems import detect_anomalies
from models.sentiment import aggregate_sentiment_per_game
from models.topic_model import build_review_topics_pipeline


def _build_features(
    games_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    sent = aggregate_sentiment_per_game(reviews_df, game_id_col="gameId", text_col="review")
    english = reviews_df[reviews_df["language"].fillna("").astype(str).str.lower() == "english"]
    topics = None
    if len(english) >= 10:
        topics = build_review_topics_pipeline(english, n_topics=8, method="lda")["game_topics"]
    merged, X, _ = build_feature_matrix(games_df, sent, topics)
    return merged, X


def _labels_for_games(merged_df: pd.DataFrame, labels_df: pd.DataFrame) -> np.ndarray:
    if "id" not in merged_df.columns:
        raise ValueError("Merged feature dataframe missing 'id' column.")
    joined = merged_df[["id"]].merge(
        labels_df[["id", "is_gem"]],
        on="id",
        how="left",
    )
    y = pd.to_numeric(joined["is_gem"], errors="coerce").fillna(0).astype(int).to_numpy()
    return y


def _evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    top_k = max(1, int(len(y_score) * 0.1))
    top_idx = np.argsort(y_score)[::-1][:top_k]
    pred_top = np.zeros_like(y_true)
    pred_top[top_idx] = 1
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision_at_10pct": float(precision_score(y_true, pred_top, zero_division=0)),
        "recall_at_10pct": float(recall_score(y_true, pred_top, zero_division=0)),
    }


def _run_method(
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str,
    contamination: float,
    seeds: list[int],
) -> dict[str, object]:
    run_metrics: list[dict[str, float]] = []
    for seed in seeds:
        labels = detect_anomalies(
            X,
            contamination=contamination,
            method=method,
            random_state=seed,
        )
        # Outlier label is -1; convert to score where higher means more anomaly.
        y_score = (labels == -1).astype(float)
        run_metrics.append(_evaluate_scores(y_true, y_score))
    keys = ["pr_auc", "precision_at_10pct", "recall_at_10pct"]
    aggregate: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.asarray([m[key] for m in run_metrics], dtype=float)
        aggregate[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return {
        "method": method,
        "runs": run_metrics,
        "aggregate": aggregate,
    }


def benchmark(
    *,
    games_csv: Path,
    reviews_csv: Path,
    labels_csv: Path,
    contamination: float,
    seeds: list[int],
) -> dict[str, object]:
    games_df = pd.read_csv(games_csv)
    reviews_df = pd.read_csv(reviews_csv)
    labels_df = pd.read_csv(labels_csv)
    if "id" not in labels_df.columns or "is_gem" not in labels_df.columns:
        # training dataset-style labels
        if {"appid", "is_gem"}.issubset(labels_df.columns):
            labels_df = labels_df.rename(columns={"appid": "id"})
        else:
            raise ValueError("labels CSV must include id/is_gem or appid/is_gem.")

    merged_df, X = _build_features(games_df, reviews_df)
    y_true = _labels_for_games(merged_df, labels_df)
    methods = ["isolation_forest", "lof", "one_class_svm"]
    results = [
        _run_method(X, y_true, method=method, contamination=contamination, seeds=seeds)
        for method in methods
    ]
    winner = max(results, key=lambda r: r["aggregate"]["pr_auc"]["mean"])
    return {
        "meta": {
            "n_games": int(len(merged_df)),
            "n_labeled_positive": int((y_true == 1).sum()),
            "n_labeled_negative": int((y_true == 0).sum()),
            "contamination": float(contamination),
            "seeds": seeds,
        },
        "results": results,
        "winner": {
            "method": winner["method"],
            "pr_auc_mean": winner["aggregate"]["pr_auc"]["mean"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark anomaly methods against gem labels.")
    ap.add_argument("--games-csv", default=local_data_file("steam_games_clean.csv"))
    ap.add_argument("--reviews-csv", default=local_data_file("steam_reviews_clean.csv"))
    ap.add_argument("--labels-csv", default=local_data_file("training_dataset.csv"))
    ap.add_argument("--contamination", type=float, default=0.1)
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--out", default=model_artifact_file("anomaly_benchmark.json"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    report = benchmark(
        games_csv=Path(args.games_csv),
        reviews_csv=Path(args.reviews_csv),
        labels_csv=Path(args.labels_csv),
        contamination=args.contamination,
        seeds=seeds,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved anomaly benchmark report to {out}")
    print(f"Winner method: {report['winner']['method']}")


if __name__ == "__main__":
    main()

