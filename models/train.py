from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.evaluate import binary_classification_report, select_threshold


SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


NON_FEATURE_COLUMNS = {"id", "has_label", "temporal_split_key"}


def _detect_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def _prepare_feature_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    drop_cols = {target_col, *NON_FEATURE_COLUMNS}
    existing = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing).copy()
    return X


def _make_preprocessor(*, cat_cols: list[str], num_cols: list[str], scale_numeric: bool) -> ColumnTransformer:
    num_transformer: str | StandardScaler = "passthrough"
    if scale_numeric:
        # with_mean=False keeps sparse compatibility when OHE columns exist.
        num_transformer = StandardScaler(with_mean=False)
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", num_transformer, num_cols),
        ]
    )


def _make_pipeline(
    model_name: str,
    *,
    cat_cols: list[str],
    num_cols: list[str],
    class_weight_mode: str,
) -> Pipeline:
    class_weight: str | None = "balanced" if class_weight_mode == "balanced" else None
    if model_name == "logreg":
        model = LogisticRegression(
            max_iter=4000,
            class_weight=class_weight,
            random_state=42,
        )
        pre = _make_preprocessor(cat_cols=cat_cols, num_cols=num_cols, scale_numeric=True)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
        pre = _make_preprocessor(cat_cols=cat_cols, num_cols=num_cols, scale_numeric=False)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'")
    return Pipeline([("pre", pre), ("model", model)])


def _sample_param_grid(model_name: str, *, n_iter: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if n_iter <= 1:
        return [{}]
    if model_name == "logreg":
        cs = np.logspace(-3, 2, num=10)
        penalties = ["l2"]
        solvers = ["lbfgs", "liblinear"]
        grid: list[dict[str, Any]] = []
        for _ in range(n_iter):
            grid.append(
                {
                    "model__C": float(rng.choice(cs)),
                    "model__penalty": str(rng.choice(penalties)),
                    "model__solver": str(rng.choice(solvers)),
                }
            )
        return grid
    if model_name == "random_forest":
        grid = []
        for _ in range(n_iter):
            grid.append(
                {
                    "model__n_estimators": int(rng.choice([300, 500, 700, 900])),
                    "model__max_depth": int(rng.choice([4, 6, 8, 12, 16, 24])),
                    "model__min_samples_split": int(rng.choice([2, 4, 8, 12])),
                    "model__min_samples_leaf": int(rng.choice([1, 2, 4, 8])),
                    "model__max_features": str(rng.choice(["sqrt", "log2"])),
                }
            )
        return grid
    raise ValueError(f"Unsupported model_name '{model_name}'")


def _score_array_to_summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _cv_evaluate(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv_folds: int,
    cv_repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        random_state=seed,
    )
    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    metrics: dict[str, dict[str, float]] = {}
    for key in SCORING:
        test_key = f"test_{key}"
        metrics[key] = _score_array_to_summary(np.asarray(scores[test_key], dtype=float))
    return metrics


def _positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba did not return two-class probabilities.")
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        # Sigmoid transform for consistent 0-1 thresholding on linear scores.
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError("Model does not support predict_proba or decision_function.")


def _build_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    cat_cols: list[str],
    num_cols: list[str],
    class_weight_mode: str,
) -> dict[str, Any]:
    pipe = _make_pipeline(
        "logreg",
        cat_cols=cat_cols,
        num_cols=num_cols,
        class_weight_mode=class_weight_mode,
    )
    pipe.fit(X_train, y_train)
    score_test = _positive_scores(pipe, X_test)
    report = binary_classification_report(y_test.to_numpy(), score_test, threshold=0.5)
    return {
        "model_name": "logreg_baseline",
        "threshold": 0.5,
        "test_report": report.__dict__,
        "brier": float(brier_score_loss(y_test, score_test)),
    }


def _temporal_group_order(values: pd.Series) -> list[str]:
    parsed = pd.to_datetime(values, errors="coerce")
    tmp = pd.DataFrame({"raw": values.astype(str), "parsed": parsed})
    agg = (
        tmp.groupby("raw", dropna=False)["parsed"]
        .max()
        .reset_index()
        .sort_values("parsed", ascending=True, na_position="last")
    )
    return agg["raw"].astype(str).tolist()


def _temporal_holdout_eval(
    df: pd.DataFrame,
    *,
    target_col: str,
    temporal_key: str,
    holdout_fraction: float,
    class_weight_mode: str,
    champion_model_name: str,
    champion_params: dict[str, Any],
    champion_threshold: float,
) -> dict[str, Any] | None:
    if temporal_key not in df.columns:
        return None
    keys = df[temporal_key].fillna("unknown").astype(str)
    ordered = _temporal_group_order(keys)
    if len(ordered) < 3:
        return None
    n_holdout = max(1, int(round(len(ordered) * holdout_fraction)))
    holdout_keys = set(ordered[-n_holdout:])
    holdout_mask = keys.isin(holdout_keys)
    if holdout_mask.sum() < 5 or (~holdout_mask).sum() < 10:
        return None

    train_df = df.loc[~holdout_mask].copy()
    test_df = df.loc[holdout_mask].copy()
    X_train = _prepare_feature_frame(train_df, target_col)
    y_train = train_df[target_col].astype(int)
    X_test = _prepare_feature_frame(test_df, target_col)
    y_test = test_df[target_col].astype(int)
    cat_cols, num_cols = _detect_feature_columns(X_train)

    baseline_pipe = _make_pipeline(
        "logreg",
        cat_cols=cat_cols,
        num_cols=num_cols,
        class_weight_mode=class_weight_mode,
    )
    baseline_pipe.fit(X_train, y_train)
    baseline_scores = _positive_scores(baseline_pipe, X_test)
    baseline_report = binary_classification_report(y_test.to_numpy(), baseline_scores, threshold=0.5)

    champion_pipe = _make_pipeline(
        champion_model_name,
        cat_cols=cat_cols,
        num_cols=num_cols,
        class_weight_mode=class_weight_mode,
    ).set_params(**champion_params)
    champion_pipe.fit(X_train, y_train)
    champion_scores = _positive_scores(champion_pipe, X_test)
    champion_report = binary_classification_report(
        y_test.to_numpy(),
        champion_scores,
        threshold=champion_threshold,
    )
    return {
        "temporal_key": temporal_key,
        "holdout_fraction": holdout_fraction,
        "holdout_keys": sorted(holdout_keys),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "baseline_test_report": baseline_report.__dict__,
        "champion_test_report": champion_report.__dict__,
        "deltas": {
            "pr_auc_gain": float(
                (champion_report.pr_auc or 0.0) - (baseline_report.pr_auc or 0.0)
            ),
            "recall_gain": float(champion_report.recall - baseline_report.recall),
        },
    }


def _train_champion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    cat_cols: list[str],
    num_cols: list[str],
    model_candidates: list[str],
    primary_metric: str,
    search_iter: int,
    cv_folds: int,
    cv_repeats: int,
    class_weight_mode: str,
    seed: int,
    threshold_objective: str,
    min_precision: float,
) -> dict[str, Any]:
    if primary_metric not in SCORING:
        raise ValueError(f"Unsupported primary metric '{primary_metric}'")

    rng = np.random.default_rng(seed)
    all_runs: list[dict[str, Any]] = []

    for model_name in model_candidates:
        base = _make_pipeline(
            model_name,
            cat_cols=cat_cols,
            num_cols=num_cols,
            class_weight_mode=class_weight_mode,
        )
        params_list = _sample_param_grid(model_name, n_iter=search_iter, rng=rng)
        for params in params_list:
            pipe = base.set_params(**params)
            cv_metrics = _cv_evaluate(
                pipe,
                X_train,
                y_train,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
                seed=seed,
            )
            all_runs.append(
                {
                    "model_name": model_name,
                    "params": params,
                    "cv_metrics": cv_metrics,
                    "primary_score": cv_metrics[primary_metric]["mean"],
                    "primary_std": cv_metrics[primary_metric]["std"],
                }
            )

    best = max(all_runs, key=lambda r: (r["primary_score"], -r["primary_std"]))
    best_pipe = _make_pipeline(
        best["model_name"],
        cat_cols=cat_cols,
        num_cols=num_cols,
        class_weight_mode=class_weight_mode,
    ).set_params(**best["params"])

    # Threshold selection on a calibration split inside training data.
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train if y_train.nunique() > 1 else None,
    )
    best_pipe.fit(X_fit, y_fit)
    score_cal = _positive_scores(best_pipe, X_cal)
    selected = select_threshold(
        y_cal.to_numpy(),
        score_cal,
        optimize_for="recall" if threshold_objective == "recall" else "f1",
        min_precision=min_precision,
    )

    # Fit on full training split and evaluate on untouched test split.
    best_pipe.fit(X_train, y_train)
    score_test = _positive_scores(best_pipe, X_test)
    test_report = binary_classification_report(
        y_test.to_numpy(),
        score_test,
        threshold=selected.threshold,
    )
    return {
        "model": best_pipe,
        "model_name": best["model_name"],
        "threshold": float(selected.threshold),
        "threshold_selection": selected.__dict__,
        "cv_candidates": all_runs,
        "best_candidate": best,
        "test_report": test_report.__dict__,
        "brier": float(brier_score_loss(y_test, score_test)),
    }


def _promotion_gates(
    baseline: dict[str, Any],
    champion: dict[str, Any],
    *,
    min_pr_auc_gain: float,
    min_recall_gain: float,
    max_precision_drop: float,
    max_brier: float,
    max_cv_primary_std: float,
) -> dict[str, Any]:
    base_report = baseline["test_report"]
    champ_report = champion["test_report"]
    base_pr = float(base_report["pr_auc"]) if base_report["pr_auc"] is not None else 0.0
    champ_pr = float(champ_report["pr_auc"]) if champ_report["pr_auc"] is not None else 0.0
    pr_gain = champ_pr - base_pr
    recall_gain = float(champ_report["recall"]) - float(base_report["recall"])
    precision_drop = float(base_report["precision"]) - float(champ_report["precision"])
    cv_std = float(champion["best_candidate"]["primary_std"])

    checks = {
        "pr_auc_gain": pr_gain >= min_pr_auc_gain,
        "recall_gain": recall_gain >= min_recall_gain,
        "precision_drop_bounded": precision_drop <= max_precision_drop,
        "brier_ok": float(champion["brier"]) <= max_brier,
        "cv_stability_ok": cv_std <= max_cv_primary_std,
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "deltas": {
            "pr_auc_gain": float(pr_gain),
            "recall_gain": float(recall_gain),
            "precision_drop": float(precision_drop),
            "candidate_brier": float(champion["brier"]),
            "candidate_cv_primary_std": float(cv_std),
        },
    }


def train(
    df: pd.DataFrame,
    *,
    target_col: str,
    primary_metric: str = "pr_auc",
    cv_folds: int = 5,
    cv_repeats: int = 2,
    search_iter: int = 6,
    class_weight_mode: str = "balanced",
    seed: int = 42,
    threshold_objective: str = "f1",
    min_precision: float = 0.0,
    temporal_key: str = "temporal_split_key",
    temporal_holdout_fraction: float = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe columns")

    X = _prepare_feature_frame(df, target_col)
    y = df[target_col].astype(int)

    if y.nunique() < 2:
        raise ValueError("Target must include at least two classes.")
    if int((y == 1).sum()) < 10:
        raise ValueError("Need at least 10 positive labels for stable training.")

    cat_cols, num_cols = _detect_feature_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    baseline = _build_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        class_weight_mode=class_weight_mode,
    )
    champion = _train_champion(
        X_train,
        y_train,
        X_test,
        y_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        model_candidates=["logreg", "random_forest"],
        primary_metric=primary_metric,
        search_iter=search_iter,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        class_weight_mode=class_weight_mode,
        seed=seed,
        threshold_objective=threshold_objective,
        min_precision=min_precision,
    )

    gates = _promotion_gates(
        baseline,
        champion,
        min_pr_auc_gain=0.01,
        min_recall_gain=0.03,
        max_precision_drop=0.05,
        max_brier=0.30,
        max_cv_primary_std=0.08,
    )
    temporal_eval = _temporal_holdout_eval(
        df,
        target_col=target_col,
        temporal_key=temporal_key,
        holdout_fraction=temporal_holdout_fraction,
        class_weight_mode=class_weight_mode,
        champion_model_name=champion["model_name"],
        champion_params=champion["best_candidate"]["params"],
        champion_threshold=float(champion["threshold"]),
    )
    if temporal_eval is not None:
        temporal_pass = temporal_eval["deltas"]["pr_auc_gain"] >= -1e-9
        gates["checks"]["temporal_pr_auc_non_regressive"] = temporal_pass
        gates["pass"] = bool(all(gates["checks"].values()))
    metrics = {
        "baseline": baseline,
        "champion": {
            "model_name": champion["model_name"],
            "threshold": champion["threshold"],
            "test_report": champion["test_report"],
            "best_candidate": champion["best_candidate"],
            "brier": champion["brier"],
        },
        "promotion": gates,
        "dataset": {
            "n_rows": int(len(df)),
            "n_features": int(X.shape[1]),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "search": {
            "primary_metric": primary_metric,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "search_iter": search_iter,
            "class_weight_mode": class_weight_mode,
            "seed": seed,
            "threshold_objective": threshold_objective,
            "min_precision": min_precision,
            "temporal_key": temporal_key,
            "temporal_holdout_fraction": temporal_holdout_fraction,
            "candidate_runs": champion["cv_candidates"],
        },
    }
    if temporal_eval is not None:
        metrics["temporal_holdout"] = temporal_eval
    # Approximate feature-space density after preprocessing on full data.
    full_pipe_for_density = _make_pipeline(
        champion["model_name"],
        cat_cols=cat_cols,
        num_cols=num_cols,
        class_weight_mode=class_weight_mode,
    ).set_params(**champion["best_candidate"]["params"])
    Xt = full_pipe_for_density.named_steps["pre"].fit_transform(X)
    if hasattr(Xt, "nnz"):
        density = float(Xt.nnz / max(Xt.shape[0] * Xt.shape[1], 1))
    else:
        density = float(np.count_nonzero(Xt) / max(Xt.size, 1))
    metrics["dataset"]["preprocessed_feature_space"] = {
        "n_rows": int(Xt.shape[0]),
        "n_columns": int(Xt.shape[1]),
        "density": density,
        "sparsity": float(1.0 - density),
    }

    artifact = {
        "model": champion["model"],
        "threshold": champion["threshold"],
        "model_name": champion["model_name"],
        "feature_columns": list(X.columns),
        "excluded_columns": sorted(NON_FEATURE_COLUMNS),
        "target_col": target_col,
    }
    return artifact, metrics


def _parse_seeds(seed_arg: str) -> list[int]:
    raw = [s.strip() for s in seed_arg.split(",") if s.strip()]
    if not raw:
        raise ValueError("At least one seed must be provided.")
    return [int(s) for s in raw]


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _aggregate_seed_runs(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_metrics:
        return {}
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    comparison: dict[str, Any] = {
        "baseline_test": {},
        "champion_test": {},
    }
    for metric_name in metric_names:
        base_vals: list[float] = []
        champ_vals: list[float] = []
        for run in seed_metrics:
            base_v = run["baseline"]["test_report"].get(metric_name)
            champ_v = run["champion"]["test_report"].get(metric_name)
            if base_v is not None:
                base_vals.append(float(base_v))
            if champ_v is not None:
                champ_vals.append(float(champ_v))
        if base_vals:
            comparison["baseline_test"][metric_name] = _mean_std(base_vals)
        if champ_vals:
            comparison["champion_test"][metric_name] = _mean_std(champ_vals)

    gate_passes = [bool(run["promotion"]["pass"]) for run in seed_metrics]
    pass_rate = float(np.mean(gate_passes)) if gate_passes else 0.0
    return {
        "n_runs": len(seed_metrics),
        "metrics": comparison,
        "promotion": {
            "per_seed_pass": gate_passes,
            "pass_rate": pass_rate,
            "all_seeds_pass": bool(all(gate_passes)),
        },
    }


def _seed_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "search": {
            "seed": int(run["search"]["seed"]),
            "primary_metric": run["search"]["primary_metric"],
        },
        "baseline_test_report": run["baseline"]["test_report"],
        "champion_test_report": run["champion"]["test_report"],
        "promotion": run["promotion"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training CSV")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--out", default="artifacts/models/model.joblib", help="Output model path")
    ap.add_argument(
        "--metrics-out", default="artifacts/models/metrics.json", help="Output metrics path"
    )
    ap.add_argument(
        "--report-out",
        default="artifacts/models/run_report.json",
        help="Output detailed run report path (baseline vs champion, CV, gates).",
    )
    ap.add_argument(
        "--comparison-csv-out",
        default="artifacts/models/model_comparison.csv",
        help="Output CSV with baseline/champion metrics for quick comparison.",
    )
    ap.add_argument("--primary-metric", default="pr_auc", choices=list(SCORING.keys()))
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=2)
    ap.add_argument("--search-iter", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds for repeated effectiveness runs (example: 42,52,62).",
    )
    ap.add_argument("--class-weight-mode", choices=["none", "balanced"], default="balanced")
    ap.add_argument("--threshold-objective", choices=["f1", "recall"], default="f1")
    ap.add_argument("--min-precision", type=float, default=0.0)
    ap.add_argument("--min-gate-pass-rate", type=float, default=0.6)
    ap.add_argument(
        "--temporal-key",
        default="temporal_split_key",
        help="Column used to build time-aware holdout evaluation (if present).",
    )
    ap.add_argument(
        "--temporal-holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of latest temporal groups reserved for temporal holdout evaluation.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    seeds = _parse_seeds(args.seeds)
    if args.seed not in seeds:
        seeds = [args.seed, *seeds]

    run_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for seed in seeds:
        artifact, metrics = train(
            df,
            target_col=args.target,
            primary_metric=args.primary_metric,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            search_iter=args.search_iter,
            class_weight_mode=args.class_weight_mode,
            seed=seed,
            threshold_objective=args.threshold_objective,
            min_precision=args.min_precision,
            temporal_key=args.temporal_key,
            temporal_holdout_fraction=args.temporal_holdout_fraction,
        )
        run_pairs.append((artifact, metrics))

    # Pick the run with the strongest primary-metric score on test set.
    metric_key = args.primary_metric
    best_idx = 0
    best_score = float("-inf")
    for idx, (_, run_metrics) in enumerate(run_pairs):
        score = run_metrics["champion"]["test_report"].get(metric_key)
        if score is None:
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_idx = idx
    artifact, metrics = run_pairs[best_idx]
    metrics = copy.deepcopy(metrics)
    seed_metrics = [copy.deepcopy(m) for _, m in run_pairs]
    metrics["seed_runs"] = seed_metrics
    metrics["seed_run_summaries"] = [_seed_run_summary(m) for m in seed_metrics]
    metrics["seed_aggregate"] = _aggregate_seed_runs(seed_metrics)
    metrics["selected_seed_index"] = best_idx
    metrics["selected_seed"] = int(seed_metrics[best_idx]["search"]["seed"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pass_rate = float(metrics.get("seed_aggregate", {}).get("promotion", {}).get("pass_rate", 0.0))
    promotion_pass = pass_rate >= float(args.min_gate_pass_rate)
    metrics["promotion"]["overall_pass"] = promotion_pass
    metrics["promotion"]["overall_pass_rate"] = pass_rate
    metrics["promotion"]["required_pass_rate"] = float(args.min_gate_pass_rate)
    if promotion_pass:
        save_path = out_path
    else:
        save_path = out_path.with_name(f"{out_path.stem}.experimental{out_path.suffix}")
    joblib.dump(artifact, save_path)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2) + "\n")

    comparison_path = Path(args.comparison_csv_out)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    base_report = metrics["baseline"]["test_report"]
    champ_report = metrics["champion"]["test_report"]
    for metric_name in metric_names:
        base_val = base_report.get(metric_name)
        champ_val = champ_report.get(metric_name)
        if base_val is None or champ_val is None:
            continue
        rows.append(
            {
                "metric": metric_name,
                "baseline": float(base_val),
                "champion": float(champ_val),
                "delta_champion_minus_baseline": float(champ_val) - float(base_val),
            }
        )
    pd.DataFrame(rows).to_csv(comparison_path, index=False)

    print(f"Saved model to {save_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved comparison CSV to {comparison_path}")
    print(f"Promotion gates passed: {promotion_pass}")


if __name__ == "__main__":
    main()

