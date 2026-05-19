"""Clustering pipeline: feature engineering, clustering algorithms, and
dimensionality reduction for Steam game data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "owners_min",
    "owners_max",
    "average_forever",
    "median_forever",
    "metacritic",
    "required_age",
    "reviewTotalPositive",
    "reviewTotalNegative",
    "reviewTotalReviews",
    "reviewScore",
]

BOOL_FEATURES = ["is_free", "windows", "mac", "linux"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(
    games_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
    topics_df: pd.DataFrame | None = None,
    game_id_col: str = "id",
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Merge game metadata with NLP features and return a scaled matrix.

    Returns (merged_df, X_scaled, feature_names).
    """
    df = games_df.copy()

    if sentiment_df is not None:
        df = df.merge(sentiment_df, left_on=game_id_col, right_on="gameId", how="left")
    if topics_df is not None:
        df = df.merge(topics_df, left_on=game_id_col, right_on="gameId", how="left")

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    bool_cols = [c for c in BOOL_FEATURES if c in df.columns]
    sentiment_cols = [
        c for c in df.columns
        if c.startswith("sentiment_") or c == "avg_review_length"
    ]
    sentiment_cols = [
        c for c in sentiment_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    topic_cols = [c for c in df.columns if c.startswith("topic_")]

    all_cols = num_cols + bool_cols + sentiment_cols + topic_cols
    feature_df = df[all_cols].copy()

    for c in bool_cols:
        feature_df[c] = feature_df[c].astype(int)

    feature_df = feature_df.fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

    return df, X, all_cols


# ---------------------------------------------------------------------------
# Clustering algorithms
# ---------------------------------------------------------------------------

def run_kmeans(
    X: np.ndarray,
    k_range: range | None = None,
) -> dict[str, Any]:
    """K-Means over a range of k values.  Returns inertias, silhouettes,
    and per-k models/labels."""
    n_samples = X.shape[0]
    if k_range is None:
        k_range = range(2, min(11, n_samples))

    inertias: list[float] = []
    silhouettes: list[float] = []
    models: dict[int, dict] = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
        silhouettes.append(sil)
        models[k] = {"model": km, "labels": labels}

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "models": models,
    }


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> dict[str, Any]:
    """DBSCAN clustering. Noise points (label -1) may be hidden-gem
    candidates."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    sil = silhouette_score(X, labels) if n_clusters > 1 else -1.0
    return {
        "model": db,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": sil,
    }


def run_hierarchical(
    X: np.ndarray,
    n_clusters: int = 5,
) -> dict[str, Any]:
    """Agglomerative (Ward) clustering."""
    n_clusters = min(n_clusters, X.shape[0] - 1) if X.shape[0] > 2 else 2
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
    return {"model": hc, "labels": labels, "silhouette": sil}


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_pca(
    X: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, PCA]:
    """PCA projection (also useful as pre-processing before clustering)."""
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=42)
    return pca.fit_transform(X), pca


def reduce_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
) -> tuple[np.ndarray, Any]:
    """UMAP projection for visualisation."""
    import umap  # optional heavy dependency

    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=max(n_neighbors, 2),
        random_state=42,
    )
    return reducer.fit_transform(X), reducer
