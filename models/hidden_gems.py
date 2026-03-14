"""Hidden-gem scoring: surface high-quality but low-visibility games."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def _safe_scale(values: np.ndarray) -> np.ndarray:
    """MinMax-scale a 1-D array, handling constant inputs gracefully."""
    arr = values.reshape(-1, 1)
    scaler = MinMaxScaler()
    return scaler.fit_transform(arr).ravel()


def compute_quality_score(games_df: pd.DataFrame) -> pd.Series:
    """Composite quality signal (0-1) from reviews, sentiment, playtime."""
    score = np.zeros(len(games_df))

    total = games_df.get("reviewTotalReviews", pd.Series(0, index=games_df.index))
    positive = games_df.get("reviewTotalPositive", pd.Series(0, index=games_df.index))
    ratio = np.where(total > 0, positive / total, 0.5)
    score += _safe_scale(ratio) * 0.35

    if "sentiment_mean" in games_df.columns:
        sent = games_df["sentiment_mean"].fillna(0).values
        score += _safe_scale(sent) * 0.30

    if "average_forever" in games_df.columns:
        pt = np.log1p(games_df["average_forever"].fillna(0).values)
        score += _safe_scale(pt) * 0.20

    if "reviewScore" in games_df.columns:
        rs = games_df["reviewScore"].fillna(0).values
        score += _safe_scale(rs) * 0.15

    return pd.Series(score, index=games_df.index)


def compute_visibility_score(games_df: pd.DataFrame) -> pd.Series:
    """Inverse-visibility (0 = very visible, 1 = very hidden)."""
    score = np.zeros(len(games_df))

    if "owners_max" in games_df.columns:
        owners = np.log1p(games_df["owners_max"].fillna(0).values)
        score += (1 - _safe_scale(owners)) * 0.40

    if "reviewTotalReviews" in games_df.columns:
        reviews = np.log1p(games_df["reviewTotalReviews"].fillna(0).values)
        score += (1 - _safe_scale(reviews)) * 0.35

    if "metacritic" in games_df.columns:
        no_meta = games_df["metacritic"].isna().astype(float).values
        score += no_meta * 0.25

    return pd.Series(score, index=games_df.index)


def compute_hidden_gem_score(games_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with quality, visibility, and overall hidden-gem
    scores sorted by the combined score descending."""
    quality = compute_quality_score(games_df)
    visibility = compute_visibility_score(games_df)
    combined = quality * 0.6 + visibility * 0.4

    cols = ["id"]
    if "name" in games_df.columns:
        cols.append("name")
    result = games_df[cols].copy()
    result["quality_score"] = quality.values
    result["visibility_score"] = visibility.values
    result["hidden_gem_score"] = combined.values
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
