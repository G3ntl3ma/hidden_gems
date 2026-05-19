from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from scripts._runtime import bootstrap_project_root, local_data_file, model_artifact_file

bootstrap_project_root()

from models.hidden_gems import HiddenGemWeightProfile, compute_hidden_gem_score, default_weight_profile


def _sample_simplex(keys: list[str], rng: np.random.Generator) -> dict[str, float]:
    if not keys:
        return {}
    vals = rng.dirichlet(np.ones(len(keys)))
    return {k: float(v) for k, v in zip(keys, vals, strict=False)}


def _score_profile(df: pd.DataFrame, y: pd.Series, profile: HiddenGemWeightProfile) -> dict[str, float]:
    scored = compute_hidden_gem_score(df, profile=profile)
    y_score = scored["hidden_gem_score"].to_numpy(dtype=float)
    y_true = y.to_numpy(dtype=int)
    ap = float(average_precision_score(y_true, y_score))
    top_k = max(1, int(len(scored) * 0.1))
    top_idx = np.argsort(y_score)[::-1][:top_k]
    top_pred = np.zeros_like(y_true)
    top_pred[top_idx] = 1
    precision_at_k = float(precision_score(y_true, top_pred, zero_division=0))
    recall_at_k = float(recall_score(y_true, top_pred, zero_division=0))
    return {
        "pr_auc": ap,
        "precision_at_10pct": precision_at_k,
        "recall_at_10pct": recall_at_k,
    }


def _profile_to_dict(profile: HiddenGemWeightProfile) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "quality_core_weights": profile.quality_core_weights,
        "longevity_weights": profile.longevity_weights,
        "visibility_weights": profile.visibility_weights,
        "quality_longevity_mix": profile.quality_longevity_mix,
        "hidden_gem_mix": profile.hidden_gem_mix,
    }


def _sample_profile(base: HiddenGemWeightProfile, rng: np.random.Generator) -> HiddenGemWeightProfile:
    return HiddenGemWeightProfile(
        quality_core_weights=_sample_simplex(list(base.quality_core_weights), rng),
        longevity_weights=_sample_simplex(list(base.longevity_weights), rng),
        visibility_weights=_sample_simplex(list(base.visibility_weights), rng),
        quality_longevity_mix=_sample_simplex(list(base.quality_longevity_mix), rng),
        hidden_gem_mix=_sample_simplex(list(base.hidden_gem_mix), rng),
        profile_name="learned_random_search",
    )


def learn_profile(
    df: pd.DataFrame,
    target_col: str,
    *,
    trials: int,
    seed: int,
    val_size: float,
) -> dict[str, object]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    feature_df = df.copy()
    rng = np.random.default_rng(seed)

    train_df, val_df, y_train, y_val = train_test_split(
        feature_df,
        y,
        test_size=val_size,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    base = default_weight_profile()
    baseline_metrics = _score_profile(val_df, y_val, base)
    best_profile = base
    best_metrics = baseline_metrics
    best_score = baseline_metrics["pr_auc"]

    for _ in range(max(trials, 1)):
        candidate = _sample_profile(base, rng)
        train_metrics = _score_profile(train_df, y_train, candidate)
        if train_metrics["pr_auc"] > best_score:
            best_profile = candidate
            best_score = train_metrics["pr_auc"]
            best_metrics = _score_profile(val_df, y_val, candidate)

    best_profile = HiddenGemWeightProfile(
        quality_core_weights=best_profile.quality_core_weights,
        longevity_weights=best_profile.longevity_weights,
        visibility_weights=best_profile.visibility_weights,
        quality_longevity_mix=best_profile.quality_longevity_mix,
        hidden_gem_mix=best_profile.hidden_gem_mix,
        profile_name="learned_from_labels",
    )

    return {
        "baseline_validation": baseline_metrics,
        "learned_validation": best_metrics,
        "improvements": {
            "pr_auc_gain": float(best_metrics["pr_auc"] - baseline_metrics["pr_auc"]),
            "precision_at_10pct_gain": float(
                best_metrics["precision_at_10pct"] - baseline_metrics["precision_at_10pct"]
            ),
            "recall_at_10pct_gain": float(
                best_metrics["recall_at_10pct"] - baseline_metrics["recall_at_10pct"]
            ),
        },
        "profile": _profile_to_dict(best_profile),
        "meta": {
            "trials": int(max(trials, 1)),
            "seed": int(seed),
            "validation_size": float(val_size),
            "n_rows": int(len(df)),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Learn hidden-gem weights from labeled data.")
    ap.add_argument("--csv", default=local_data_file("training_dataset.csv"), help="Training dataset CSV with labels.")
    ap.add_argument("--target", default="is_gem", help="Binary label column.")
    ap.add_argument("--trials", type=int, default=400, help="Random-search trials.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--val-size", type=float, default=0.25, help="Validation split fraction.")
    ap.add_argument(
        "--weights-out",
        default=model_artifact_file("hidden_gem_weights.json"),
        help="Learned weight profile output path.",
    )
    ap.add_argument(
        "--report-out",
        default=model_artifact_file("hidden_gem_weight_report.json"),
        help="Evaluation report output path.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    result = learn_profile(
        df,
        args.target,
        trials=args.trials,
        seed=args.seed,
        val_size=args.val_size,
    )

    weights_path = Path(args.weights_out)
    report_path = Path(args.report_out)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(result["profile"], indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Saved learned weights to {weights_path}")
    print(f"Saved weight-learning report to {report_path}")


if __name__ == "__main__":
    main()

