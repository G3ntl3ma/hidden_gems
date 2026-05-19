"""Persistent precomputed artifacts for analysis dashboard workloads."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.clustering import build_feature_matrix
from models.hidden_gems import (
    compute_hidden_gem_score,
    default_weight_profile,
    detect_anomalies,
    load_weight_profile,
)
from models.sentiment import (
    DEFAULT_SENTIMENT_BACKEND,
    DEFAULT_SENTIMENT_MODEL,
    add_sentiment_to_reviews,
    aggregate_sentiment_per_game,
)
from models.topic_model import TOPIC_EXTRA_STOP_WORDS, build_review_topics_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ANALYSIS_CACHE_DIR = ARTIFACTS_DIR / "analysis"
HIDDEN_GEM_WEIGHT_PATH = ARTIFACTS_DIR / "models" / "hidden_gem_weights.json"


DEFAULT_TOPIC_COUNT = 8
DEFAULT_TOPIC_METHOD = os.getenv("TOPIC_METHOD", "lda")
DEFAULT_TOPIC_MAX_FEATURES = 5000
DEFAULT_ANOMALY_CONTAMINATION = 0.1
DEFAULT_HIDDEN_GEM_SCORING_VERSION = "longevity_v3"
DEFAULT_SENTIMENT_BACKEND_NAME = DEFAULT_SENTIMENT_BACKEND
DEFAULT_SENTIMENT_MODEL_NAME = DEFAULT_SENTIMENT_MODEL
DEFAULT_ANOMALY_METHOD = os.getenv("ANOMALY_METHOD", "isolation_forest").strip().lower()
DEFAULT_HIDDEN_GEM_WEIGHT_MODE = os.getenv("HIDDEN_GEM_WEIGHT_MODE", "hand_tuned").strip().lower()
DEFAULT_HIDDEN_GEM_WEIGHT_PATH = os.getenv(
    "HIDDEN_GEM_WEIGHT_PATH",
    str(HIDDEN_GEM_WEIGHT_PATH),
).strip()


ARTIFACT_FILENAMES = {
    "reviews_sentiment": "reviews_sentiment.pkl",
    "sentiment_per_game": "sentiment_per_game.pkl",
    "topic_result": "topic_result.pkl",
    "features": "features.pkl",
    "gem_scores": "gem_scores.pkl",
    "anomaly_labels": "anomaly_labels.npy",
    "manifest": "manifest.json",
}


def default_cache_dir() -> Path:
    return Path(os.getenv("ANALYSIS_CACHE_DIR", str(ANALYSIS_CACHE_DIR)))


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _required_artifact_paths(cache_dir: Path) -> list[Path]:
    return [
        cache_dir / ARTIFACT_FILENAMES["reviews_sentiment"],
        cache_dir / ARTIFACT_FILENAMES["sentiment_per_game"],
        cache_dir / ARTIFACT_FILENAMES["topic_result"],
        cache_dir / ARTIFACT_FILENAMES["features"],
        cache_dir / ARTIFACT_FILENAMES["gem_scores"],
        cache_dir / ARTIFACT_FILENAMES["anomaly_labels"],
        cache_dir / ARTIFACT_FILENAMES["manifest"],
    ]


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / ARTIFACT_FILENAMES["manifest"]


def _read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_manifest(games_csv: Path, reviews_csv: Path) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "games_csv": _source_signature(games_csv),
            "reviews_csv": _source_signature(reviews_csv),
        },
        "params": {
            "topic_count": DEFAULT_TOPIC_COUNT,
            "topic_method": DEFAULT_TOPIC_METHOD,
            "topic_max_features": DEFAULT_TOPIC_MAX_FEATURES,
            "topic_extra_stop_words": sorted(TOPIC_EXTRA_STOP_WORDS),
            "sentiment_backend": DEFAULT_SENTIMENT_BACKEND_NAME,
            "sentiment_model": DEFAULT_SENTIMENT_MODEL_NAME,
            "anomaly_method": DEFAULT_ANOMALY_METHOD,
            "anomaly_contamination": DEFAULT_ANOMALY_CONTAMINATION,
            "hidden_gem_scoring_version": DEFAULT_HIDDEN_GEM_SCORING_VERSION,
            "hidden_gem_weight_mode": DEFAULT_HIDDEN_GEM_WEIGHT_MODE,
            "hidden_gem_weight_path": DEFAULT_HIDDEN_GEM_WEIGHT_PATH,
        },
        "artifact_files": ARTIFACT_FILENAMES,
    }


def _manifest_is_fresh(cache_dir: Path, games_csv: Path, reviews_csv: Path) -> bool:
    for path in _required_artifact_paths(cache_dir):
        if not path.exists():
            return False

    manifest = _read_manifest(cache_dir)
    if manifest is None:
        return False

    expected = _build_manifest(games_csv, reviews_csv)
    inputs = manifest.get("inputs", {})
    expected_inputs = expected["inputs"]

    def _sig_compatible(current: dict[str, Any], old: dict[str, Any]) -> bool:
        # Path rewrites are allowed when source files were relocated but unchanged.
        return (
            old.get("mtime_ns") == current.get("mtime_ns")
            and old.get("size") == current.get("size")
        )

    if not _sig_compatible(expected_inputs["games_csv"], inputs.get("games_csv", {})):
        return False
    if not _sig_compatible(expected_inputs["reviews_csv"], inputs.get("reviews_csv", {})):
        return False

    params = dict(manifest.get("params", {}))
    expected_params = dict(expected["params"])

    # Legacy manifests stored the old default path. Ignore it unless learned mode uses it.
    if expected_params.get("hidden_gem_weight_mode") != "learned":
        params.pop("hidden_gem_weight_path", None)
        expected_params.pop("hidden_gem_weight_path", None)

    return params == expected_params


def _compute_artifacts(games_df: pd.DataFrame, reviews_df: pd.DataFrame) -> dict[str, Any]:
    reviews_sent = add_sentiment_to_reviews(
        reviews_df,
        backend=DEFAULT_SENTIMENT_BACKEND_NAME,
        model_name=DEFAULT_SENTIMENT_MODEL_NAME,
    )
    sent_per_game = aggregate_sentiment_per_game(
        reviews_sent,
        backend=DEFAULT_SENTIMENT_BACKEND_NAME,
        model_name=DEFAULT_SENTIMENT_MODEL_NAME,
    )

    english_reviews = reviews_df[
        reviews_df["language"].fillna("").astype(str).str.lower() == "english"
    ]
    topic_result: dict[str, Any] = {"top_words": {}, "game_topics": None}
    topic_game_df: pd.DataFrame | None = None
    if len(english_reviews) >= 10:
        raw_topic_result = build_review_topics_pipeline(
            english_reviews,
            n_topics=DEFAULT_TOPIC_COUNT,
            method=DEFAULT_TOPIC_METHOD,
            max_features=DEFAULT_TOPIC_MAX_FEATURES,
        )
        topic_game_df = raw_topic_result["game_topics"]
        topic_result = {
            "top_words": raw_topic_result["top_words"],
            "game_topics": topic_game_df,
        }

    merged_df, X, feature_names = build_feature_matrix(games_df, sent_per_game, topic_game_df)

    gems_input = games_df.copy()
    if sent_per_game is not None:
        gems_input = gems_input.merge(sent_per_game, left_on="id", right_on="gameId", how="left")
    if DEFAULT_HIDDEN_GEM_WEIGHT_MODE == "learned":
        profile = load_weight_profile(DEFAULT_HIDDEN_GEM_WEIGHT_PATH)
    else:
        profile = default_weight_profile()
    gem_scores = compute_hidden_gem_score(gems_input, profile=profile)

    anomaly_labels = np.array([], dtype=int)
    if len(games_df) >= 5:
        anomaly_labels = detect_anomalies(
            X,
            contamination=DEFAULT_ANOMALY_CONTAMINATION,
            method=DEFAULT_ANOMALY_METHOD,
        )

    return {
        "reviews_sentiment": reviews_sent,
        "sentiment_per_game": sent_per_game,
        "topic_result": topic_result,
        "features": {
            "merged_df": merged_df,
            "X": X,
            "feature_names": feature_names,
        },
        "gem_scores": gem_scores,
        "anomaly_labels": anomaly_labels,
        "anomaly_method": DEFAULT_ANOMALY_METHOD,
        "hidden_gem_weight_profile": profile.profile_name,
    }


def save_artifacts(
    artifacts: dict[str, Any],
    cache_dir: Path,
    games_csv: Path,
    reviews_csv: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(artifacts["reviews_sentiment"], cache_dir / ARTIFACT_FILENAMES["reviews_sentiment"])
    pd.to_pickle(artifacts["sentiment_per_game"], cache_dir / ARTIFACT_FILENAMES["sentiment_per_game"])
    pd.to_pickle(artifacts["topic_result"], cache_dir / ARTIFACT_FILENAMES["topic_result"])
    pd.to_pickle(artifacts["features"], cache_dir / ARTIFACT_FILENAMES["features"])
    pd.to_pickle(artifacts["gem_scores"], cache_dir / ARTIFACT_FILENAMES["gem_scores"])
    np.save(cache_dir / ARTIFACT_FILENAMES["anomaly_labels"], artifacts["anomaly_labels"])
    manifest = _build_manifest(games_csv, reviews_csv)
    (_manifest_path(cache_dir)).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_artifacts(cache_dir: Path) -> dict[str, Any]:
    return {
        "reviews_sentiment": pd.read_pickle(cache_dir / ARTIFACT_FILENAMES["reviews_sentiment"]),
        "sentiment_per_game": pd.read_pickle(cache_dir / ARTIFACT_FILENAMES["sentiment_per_game"]),
        "topic_result": pd.read_pickle(cache_dir / ARTIFACT_FILENAMES["topic_result"]),
        "features": pd.read_pickle(cache_dir / ARTIFACT_FILENAMES["features"]),
        "gem_scores": pd.read_pickle(cache_dir / ARTIFACT_FILENAMES["gem_scores"]),
        "anomaly_labels": np.load(cache_dir / ARTIFACT_FILENAMES["anomaly_labels"]),
        "manifest": _read_manifest(cache_dir),
    }


def load_or_precompute(
    games_csv: Path,
    reviews_csv: Path,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> tuple[dict[str, Any], bool, Path]:
    target_cache_dir = cache_dir or default_cache_dir()
    if not force_recompute and _manifest_is_fresh(target_cache_dir, games_csv, reviews_csv):
        return load_artifacts(target_cache_dir), True, target_cache_dir

    games_df = pd.read_csv(games_csv)
    reviews_df = pd.read_csv(reviews_csv)
    artifacts = _compute_artifacts(games_df, reviews_df)
    save_artifacts(artifacts, target_cache_dir, games_csv, reviews_csv)
    loaded = load_artifacts(target_cache_dir)
    return loaded, False, target_cache_dir
