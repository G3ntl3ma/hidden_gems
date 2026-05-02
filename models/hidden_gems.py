"""Hidden-gem scoring: surface high-quality but low-visibility games."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


QUALITY_CORE_WEIGHTS = {
    "positive_ratio": 0.28,
    "sentiment_mean": 0.22,
    "sentiment_pos_ratio": 0.08,
    "review_score": 0.15,
    "metacritic_score": 0.06,
    "lifetime_engagement": 0.11,
    "review_volume": 0.10,
}

# Blend sentiment_mean toward 0 (neutral compound) when sample review_count is below this.
SENTIMENT_SHRINK_MIN_REVIEWS = 10.0

LONGEVITY_WEIGHTS = {
    "age_signal": 0.15,
    "age_adjusted_engagement": 0.40,
    "age_adjusted_review_activity": 0.25,
    "engagement_persistence": 0.20,
}

VISIBILITY_WEIGHTS = {
    "low_review_volume": 0.45,
    "low_recent_activity": 0.20,
    "owner_hiddenness": 0.15,
    "owner_uncertainty": 0.05,
    "no_metacritic_coverage": 0.15,
}

QUALITY_LONGEVITY_MIX = {"quality_core": 0.65, "longevity": 0.35}
HIDDEN_GEM_MIX = {"quality": 0.72, "visibility": 0.28}


def _safe_scale(values: np.ndarray) -> np.ndarray:
    """MinMax-scale a 1-D array, handling constant inputs gracefully."""
    arr = values.reshape(-1, 1).astype(float)
    if len(arr) == 0:
        return np.array([], dtype=float)
    if np.isnan(arr).all():
        return np.full(len(arr), 0.5, dtype=float)
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        return np.full(len(arr), 0.5, dtype=float)
    scaler = MinMaxScaler()
    return scaler.fit_transform(arr).ravel()


def _get_numeric_column(
    games_df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> np.ndarray:
    if column not in games_df.columns:
        return np.full(len(games_df), default, dtype=float)
    values = pd.to_numeric(games_df[column], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(values, nan=default)


def _wilson_lower_bound(
    positives: np.ndarray,
    n: np.ndarray,
    z: float = 1.96,
) -> np.ndarray:
    """95% Wilson score lower bound; 0.5 when n == 0."""
    positives = np.asarray(positives, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full_like(n, 0.5, dtype=float)
    mask = n > 0
    if not np.any(mask):
        return out
    phat = positives[mask] / n[mask]
    z2 = z * z
    nm = n[mask]
    denom = 1.0 + z2 / nm
    center = phat + z2 / (2.0 * nm)
    radicand = (phat * (1.0 - phat) + z2 / (4.0 * nm)) / nm
    radicand = np.maximum(radicand, 0.0)
    margin = z * np.sqrt(radicand)
    out[mask] = np.clip((center - margin) / denom, 0.0, 1.0)
    return out


def _winsorized(values: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    lo = np.nanquantile(arr, lower)
    hi = np.nanquantile(arr, upper)
    return np.clip(arr, lo, hi)


def _weighted_blend(signals: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Blend available signals with normalized weights."""
    total = np.zeros(next(iter(signals.values())).shape[0], dtype=float)
    denom = 0.0
    for name, weight in weights.items():
        if name not in signals:
            continue
        total += np.nan_to_num(signals[name], nan=0.5) * weight
        denom += weight
    if denom <= 0:
        return np.full_like(total, 0.5, dtype=float)
    return total / denom


def _compute_quality_core(games_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reviews = _get_numeric_column(games_df, "reviewTotalReviews", default=0.0)
    positives = _get_numeric_column(games_df, "reviewTotalPositive", default=0.0)
    positive_ratio_wilson = _wilson_lower_bound(positives, reviews)
    positive_ratio = _safe_scale(_winsorized(np.clip(positive_ratio_wilson, 0.0, 1.0)))

    sentiment_mean_raw = _get_numeric_column(games_df, "sentiment_mean", default=0.0)
    if "review_count" in games_df.columns:
        rc = pd.to_numeric(games_df["review_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w_shrink = np.clip(rc / SENTIMENT_SHRINK_MIN_REVIEWS, 0.0, 1.0)
        sentiment_mean_adjusted = w_shrink * sentiment_mean_raw
    else:
        sentiment_mean_adjusted = sentiment_mean_raw
    sentiment_mean = _safe_scale(_winsorized(sentiment_mean_adjusted))
    sentiment_pos_ratio = _safe_scale(
        _winsorized(np.clip(_get_numeric_column(games_df, "sentiment_pos_ratio", default=0.5), 0.0, 1.0))
    )
    review_score = _safe_scale(_winsorized(_get_numeric_column(games_df, "reviewScore", default=0.0)))
    metacritic_score = _safe_scale(_winsorized(_get_numeric_column(games_df, "metacritic", default=0.0)))
    lifetime_engagement = _safe_scale(
        _winsorized(np.log1p(_get_numeric_column(games_df, "average_forever", default=0.0)))
    )
    review_volume = _safe_scale(_winsorized(np.log1p(reviews)))

    signals = {
        "positive_ratio": positive_ratio,
        "positive_ratio_wilson": positive_ratio_wilson,
        "sentiment_mean": sentiment_mean,
        "sentiment_pos_ratio": sentiment_pos_ratio,
        "review_score": review_score,
        "metacritic_score": metacritic_score,
        "lifetime_engagement": lifetime_engagement,
        "review_volume": review_volume,
    }
    quality_core = _weighted_blend(signals, QUALITY_CORE_WEIGHTS)
    return quality_core, signals


def _compute_longevity(games_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    release = pd.to_datetime(games_df.get("release_date"), errors="coerce")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = ((now - release).dt.total_seconds() / 86_400.0).fillna(0.0).to_numpy(dtype=float)
    age_days = np.clip(age_days, 0.0, None)

    age_signal = _safe_scale(_winsorized(np.log1p(age_days)))

    avg_forever = _get_numeric_column(games_df, "average_forever", default=0.0)
    avg_2weeks = _get_numeric_column(games_df, "average_2weeks", default=0.0)
    reviews = _get_numeric_column(games_df, "reviewTotalReviews", default=0.0)

    age_adjusted_engagement_raw = (
        np.log1p(avg_forever) * (1.0 + 0.35 * age_signal) + np.log1p(avg_2weeks) * (1.0 + 0.65 * age_signal)
    )
    age_adjusted_review_activity_raw = np.log1p(reviews) * (1.0 + 0.50 * age_signal)
    persistence_ratio_raw = avg_2weeks / (avg_forever + 1.0)

    age_adjusted_engagement = _safe_scale(_winsorized(age_adjusted_engagement_raw))
    age_adjusted_review_activity = _safe_scale(_winsorized(age_adjusted_review_activity_raw))
    engagement_persistence = _safe_scale(_winsorized(np.clip(persistence_ratio_raw, 0.0, None)))

    signals = {
        "age_signal": age_signal,
        "age_adjusted_engagement": age_adjusted_engagement,
        "age_adjusted_review_activity": age_adjusted_review_activity,
        "engagement_persistence": engagement_persistence,
    }
    longevity = _weighted_blend(signals, LONGEVITY_WEIGHTS)
    return longevity, signals


def compute_quality_score(games_df: pd.DataFrame) -> pd.Series:
    """Composite quality signal (0-1) with longevity-aware retention factors."""
    quality_core, _ = _compute_quality_core(games_df)
    longevity, _ = _compute_longevity(games_df)
    score = (
        quality_core * QUALITY_LONGEVITY_MIX["quality_core"]
        + longevity * QUALITY_LONGEVITY_MIX["longevity"]
    )
    return pd.Series(score, index=games_df.index)


def _compute_visibility(games_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build inverse-visibility signals and score."""
    reviews = _get_numeric_column(games_df, "reviewTotalReviews", default=0.0)
    low_review_volume = 1.0 - _safe_scale(_winsorized(np.log1p(reviews)))

    recent_activity = _get_numeric_column(games_df, "average_2weeks", default=0.0)
    low_recent_activity = 1.0 - _safe_scale(_winsorized(np.log1p(recent_activity)))

    owners_min = _get_numeric_column(games_df, "owners_min", default=0.0)
    owners_max = _get_numeric_column(games_df, "owners_max", default=0.0)
    owner_center = np.sqrt(np.maximum(owners_min, 0.0) * np.maximum(owners_max, 0.0))
    owner_hiddenness = 1.0 - _safe_scale(_winsorized(np.log1p(owner_center)))
    owner_uncertainty_ratio = (owners_max - owners_min) / np.maximum(owners_max, 1.0)
    owner_uncertainty = _safe_scale(_winsorized(np.clip(owner_uncertainty_ratio, 0.0, 1.0)))

    if "metacritic" in games_df.columns:
        no_metacritic_coverage = games_df["metacritic"].isna().astype(float).to_numpy(dtype=float)
    else:
        no_metacritic_coverage = np.full(len(games_df), 1.0, dtype=float)

    signals = {
        "low_review_volume": low_review_volume,
        "low_recent_activity": low_recent_activity,
        "owner_hiddenness": owner_hiddenness,
        "owner_uncertainty": owner_uncertainty,
        "no_metacritic_coverage": no_metacritic_coverage,
    }
    return _weighted_blend(signals, VISIBILITY_WEIGHTS), signals


def compute_visibility_score(games_df: pd.DataFrame) -> pd.Series:
    """Inverse-visibility (0 = very visible, 1 = very hidden)."""
    score, _ = _compute_visibility(games_df)
    return pd.Series(score, index=games_df.index)


def compute_hidden_gem_score(games_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with quality, visibility, and overall hidden-gem
    scores sorted by the combined score descending."""
    quality_core, quality_signals = _compute_quality_core(games_df)
    longevity, longevity_signals = _compute_longevity(games_df)
    quality = (
        quality_core * QUALITY_LONGEVITY_MIX["quality_core"]
        + longevity * QUALITY_LONGEVITY_MIX["longevity"]
    )

    visibility, visibility_signals = _compute_visibility(games_df)
    combined = (
        quality * HIDDEN_GEM_MIX["quality"]
        + visibility * HIDDEN_GEM_MIX["visibility"]
    )

    cols = ["id"]
    if "name" in games_df.columns:
        cols.append("name")
    result = games_df[cols].copy()
    result["quality_score"] = quality
    result["visibility_score"] = visibility
    result["hidden_gem_score"] = combined
    result["quality_core_component"] = quality_core
    result["longevity_component"] = longevity
    result["visibility_component"] = visibility

    for key, values in quality_signals.items():
        result[f"feat_quality_{key}"] = values
    for key, values in longevity_signals.items():
        result[f"feat_longevity_{key}"] = values
    for key, values in visibility_signals.items():
        result[f"feat_visibility_{key}"] = values

    return result.sort_values("hidden_gem_score", ascending=False).reset_index(drop=True)


def detect_anomalies(
    X: np.ndarray,
    contamination: float = 0.1,
) -> np.ndarray:
    """Isolation Forest anomaly detection.

    Returns an array of labels: 1 = inlier, -1 = outlier (potential gem).
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    return iso.fit_predict(X)
