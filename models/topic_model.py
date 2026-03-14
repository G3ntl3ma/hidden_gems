"""Topic modeling for Steam reviews using LDA / NMF on TF-IDF."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(
    texts: list[str],
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
) -> tuple[TfidfVectorizer, Any]:
    """Fit a TF-IDF vectorizer and return (vectorizer, tfidf_matrix)."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def fit_topic_model(
    tfidf_matrix: Any,
    n_topics: int = 10,
    method: str = "lda",
    random_state: int = 42,
) -> tuple[Any, np.ndarray]:
    """Fit LDA or NMF and return (model, topic_distributions)."""
    if method == "lda":
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=20,
        )
    elif method == "nmf":
        model = NMF(
            n_components=n_topics,
            random_state=random_state,
            max_iter=200,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'lda' or 'nmf'.")

    topic_distributions = model.fit_transform(tfidf_matrix)
    return model, topic_distributions


def get_top_words(
    model: Any,
    vectorizer: TfidfVectorizer,
    n_words: int = 10,
) -> dict[int, list[str]]:
    """Extract the top-N words for each topic."""
    names = vectorizer.get_feature_names_out()
    return {
        i: [names[j] for j in comp.argsort()[-n_words:][::-1]]
        for i, comp in enumerate(model.components_)
    }


def aggregate_topics_per_game(
    reviews_df: pd.DataFrame,
    topic_distributions: np.ndarray,
    game_id_col: str = "gameId",
) -> pd.DataFrame:
    """Average topic weights per game, plus dominant topic."""
    n_topics = topic_distributions.shape[1]
    topic_cols = [f"topic_{i}" for i in range(n_topics)]
    topic_df = pd.DataFrame(topic_distributions, columns=topic_cols)
    combined = pd.concat(
        [reviews_df[[game_id_col]].reset_index(drop=True), topic_df],
        axis=1,
    )
    agg = combined.groupby(game_id_col)[topic_cols].mean().reset_index()
    agg["dominant_topic"] = agg[topic_cols].values.argmax(axis=1)
    return agg


def build_review_topics_pipeline(
    reviews_df: pd.DataFrame,
    text_col: str = "review",
    n_topics: int = 10,
    method: str = "lda",
    max_features: int = 5000,
) -> dict[str, Any]:
    """End-to-end: TF-IDF -> topic model -> per-game aggregation.

    Returns dict with keys: vectorizer, model, topic_distributions,
    top_words, game_topics.
    """
    texts = reviews_df[text_col].fillna("").tolist()
    vectorizer, tfidf_matrix = fit_tfidf(texts, max_features=max_features)
    model, topic_distributions = fit_topic_model(
        tfidf_matrix, n_topics=n_topics, method=method,
    )
    top_words = get_top_words(model, vectorizer)
    game_topics = aggregate_topics_per_game(reviews_df, topic_distributions)
    return {
        "vectorizer": vectorizer,
        "model": model,
        "topic_distributions": topic_distributions,
        "top_words": top_words,
        "game_topics": game_topics,
    }
