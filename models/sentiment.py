"""Sentiment analysis for Steam game reviews."""

from __future__ import annotations

import os

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DEFAULT_SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "vader").strip().lower() or "vader"
DEFAULT_SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
).strip()


def _ensure_vader_lexicon() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def _analyze_sentiment_vader(texts: list[str]) -> list[dict[str, float]]:
    _ensure_vader_lexicon()
    sia = SentimentIntensityAnalyzer()
    neutral = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return [
        sia.polarity_scores(t) if isinstance(t, str) and t.strip() else neutral
        for t in texts
    ]


def _analyze_sentiment_transformer(
    texts: list[str],
    *,
    model_name: str,
) -> list[dict[str, float]]:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Transformer sentiment backend requires 'transformers'. "
            "Install it or switch backend to 'vader'."
        ) from exc

    clf = pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
    )
    neutral = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    cleaned = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    outputs = clf(cleaned, batch_size=32)
    result: list[dict[str, float]] = []
    for text, out in zip(cleaned, outputs, strict=False):
        if not text:
            result.append(neutral)
            continue
        label = str(out.get("label", "")).upper()
        score = float(out.get("score", 0.0))
        if "POS" in label:
            pos = np.clip(score, 0.0, 1.0)
            neg = 1.0 - pos
        else:
            neg = np.clip(score, 0.0, 1.0)
            pos = 1.0 - neg
        compound = float(np.clip(pos - neg, -1.0, 1.0))
        neu = float(max(0.0, 1.0 - max(pos, neg)))
        result.append(
            {
                "neg": float(neg),
                "neu": float(neu),
                "pos": float(pos),
                "compound": compound,
            }
        )
    return result


def analyze_sentiment(
    texts: list[str],
    *,
    backend: str = DEFAULT_SENTIMENT_BACKEND,
    model_name: str = DEFAULT_SENTIMENT_MODEL,
) -> list[dict[str, float]]:
    """Run sentiment on a list of texts and return neg/neu/pos/compound."""
    backend_key = backend.strip().lower()
    if backend_key == "vader":
        return _analyze_sentiment_vader(texts)
    if backend_key == "transformer":
        return _analyze_sentiment_transformer(texts, model_name=model_name)
    raise ValueError(f"Unknown sentiment backend '{backend}'. Use 'vader' or 'transformer'.")


def add_sentiment_to_reviews(
    reviews_df: pd.DataFrame,
    text_col: str = "review",
    *,
    backend: str = DEFAULT_SENTIMENT_BACKEND,
    model_name: str = DEFAULT_SENTIMENT_MODEL,
) -> pd.DataFrame:
    """Append sentiment_neg/neu/pos/compound columns to a reviews DataFrame."""
    scores = analyze_sentiment(
        reviews_df[text_col].fillna("").tolist(),
        backend=backend,
        model_name=model_name,
    )
    sentiment_df = pd.DataFrame(scores).rename(
        columns=lambda c: f"sentiment_{c}",
    )
    out = pd.concat(
        [reviews_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)],
        axis=1,
    )
    out["sentiment_backend"] = backend.strip().lower()
    return out


def aggregate_sentiment_per_game(
    reviews_df: pd.DataFrame,
    game_id_col: str = "gameId",
    text_col: str = "review",
    *,
    backend: str = DEFAULT_SENTIMENT_BACKEND,
    model_name: str = DEFAULT_SENTIMENT_MODEL,
) -> pd.DataFrame:
    """Roll up review-level sentiment to one row per game."""
    if "sentiment_compound" not in reviews_df.columns:
        reviews_df = add_sentiment_to_reviews(
            reviews_df,
            text_col=text_col,
            backend=backend,
            model_name=model_name,
        )

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
    agg["sentiment_backend"] = backend.strip().lower()
    return agg
