"""Sentiment analysis for Steam game reviews using VADER."""

from __future__ import annotations

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _ensure_vader_lexicon() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def analyze_sentiment(texts: list[str]) -> list[dict[str, float]]:
    """Run VADER sentiment on a list of texts.

    Returns one dict per text with keys: neg, neu, pos, compound.
    """
    _ensure_vader_lexicon()
    sia = SentimentIntensityAnalyzer()
    neutral = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return [
        sia.polarity_scores(t) if isinstance(t, str) and t.strip() else neutral
        for t in texts
    ]


def add_sentiment_to_reviews(
    reviews_df: pd.DataFrame,
    text_col: str = "review",
) -> pd.DataFrame:
    """Append sentiment_neg/neu/pos/compound columns to a reviews DataFrame."""
    scores = analyze_sentiment(reviews_df[text_col].fillna("").tolist())
    sentiment_df = pd.DataFrame(scores).rename(
        columns=lambda c: f"sentiment_{c}",
    )
    return pd.concat(
        [reviews_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)],
        axis=1,
    )


def aggregate_sentiment_per_game(
    reviews_df: pd.DataFrame,
    game_id_col: str = "gameId",
    text_col: str = "review",
) -> pd.DataFrame:
    """Roll up review-level sentiment to one row per game."""
    if "sentiment_compound" not in reviews_df.columns:
        reviews_df = add_sentiment_to_reviews(reviews_df, text_col=text_col)

    agg = (
        reviews_df.groupby(game_id_col)
        .agg(
            sentiment_mean=("sentiment_compound", "mean"),
            sentiment_median=("sentiment_compound", "median"),
            sentiment_std=("sentiment_compound", "std"),
            sentiment_pos_ratio=(
                "sentiment_compound",
                lambda x: (x > 0.05).mean(),
            ),
            avg_review_length=(text_col, lambda x: x.fillna("").str.len().mean()),
            review_count=(text_col, "count"),
        )
        .reset_index()
    )
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0)
    return agg
