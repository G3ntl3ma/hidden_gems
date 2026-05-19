from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._runtime import bootstrap_project_root, local_data_file, model_artifact_file

bootstrap_project_root()

from models.sentiment import aggregate_sentiment_per_game


GAME_BASE_FEATURE_COLS = [
    "id",
    "required_age",
    "is_free",
    "windows",
    "mac",
    "linux",
    "coming_soon",
    "metacritic",
    "owners_min",
    "owners_max",
    "average_2weeks",
    "average_forever",
    "median_2weeks",
    "median_forever",
    "ccu",
    "reviewNumReviews",
    "reviewScore",
    "reviewTotalPositive",
    "reviewTotalNegative",
    "reviewTotalReviews",
]


def _coerce_bool_int(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(bool).astype(int)


def _add_release_year(games_df: pd.DataFrame) -> pd.DataFrame:
    if "release_date" not in games_df.columns:
        return games_df
    df = games_df.copy()
    release = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = release.dt.year.fillna(0).astype(int)
    return df


def _add_temporal_split_key(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.copy()
    if "release_date" not in df.columns:
        df["temporal_split_key"] = "unknown"
        return df
    release = pd.to_datetime(df["release_date"], errors="coerce")
    df["temporal_split_key"] = release.dt.to_period("M").astype(str).replace("NaT", "unknown")
    return df


def _validate_label_distribution(df: pd.DataFrame, *, min_positive: int, min_positive_ratio: float) -> None:
    labeled = df.dropna(subset=["is_gem"]).copy()
    if labeled.empty:
        raise ValueError("No labeled rows found after join; cannot build supervised dataset.")
    n_pos = int((labeled["is_gem"] == 1).sum())
    n_total = int(len(labeled))
    ratio = n_pos / max(n_total, 1)
    if n_pos < min_positive:
        raise ValueError(
            f"Too few positive labels ({n_pos}); need at least {min_positive} for reliable training."
        )
    if ratio < min_positive_ratio:
        raise ValueError(
            f"Positive-label ratio too low ({ratio:.4f}); need at least {min_positive_ratio:.4f}."
        )


def _split_semicolon_values(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _add_multihot_features(
    df: pd.DataFrame,
    *,
    source_col: str,
    prefix: str,
    top_k: int,
) -> tuple[pd.DataFrame, list[str]]:
    if source_col not in df.columns:
        return df, []
    tokenized = df[source_col].map(_split_semicolon_values)
    exploded = tokenized.explode()
    if exploded.empty:
        return df, []
    top_values = (
        exploded.value_counts()
        .head(top_k)
        .index.astype(str)
        .tolist()
    )
    out = df.copy()
    new_cols: list[str] = []
    for val in top_values:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in val).strip("_")
        col = f"{prefix}__{safe}" if safe else f"{prefix}__unknown"
        if col in out.columns:
            continue
        out[col] = tokenized.map(lambda vals, needle=val: int(needle in vals))
        new_cols.append(col)
    return out, new_cols


def _add_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created: list[str] = []
    specs = [
        ("reviewScore", "sentiment_mean", "int_reviewScore_x_sentiment_mean"),
        ("metacritic", "sentiment_mean", "int_metacritic_x_sentiment_mean"),
        ("average_2weeks", "owners_min", "int_recent_engagement_per_owner_min"),
        ("reviewTotalReviews", "owners_min", "int_review_volume_per_owner_min"),
    ]
    for left, right, name in specs:
        if left not in out.columns or right not in out.columns:
            continue
        l = pd.to_numeric(out[left], errors="coerce").fillna(0.0)
        r = pd.to_numeric(out[right], errors="coerce").fillna(0.0)
        if "per_owner" in name:
            out[name] = (l / np.maximum(r, 1.0)).astype(float)
        else:
            out[name] = (l * r).astype(float)
        created.append(name)
    return out, created


def build_training_dataset(
    *,
    games_csv: Path,
    reviews_csv: Path,
    labels_csv: Path,
    require_labels: bool,
    include_temporal_split_key: bool,
    min_positive: int,
    min_positive_ratio: float,
    genre_top_k: int,
    category_top_k: int,
    include_interactions: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    games = pd.read_csv(games_csv)
    reviews = pd.read_csv(reviews_csv)
    labels = pd.read_csv(labels_csv)

    # Labels: require appid + is_gem.
    if "appid" not in labels.columns or "is_gem" not in labels.columns:
        raise ValueError("labels_csv must have columns: appid, is_gem")
    labels = labels[["appid", "is_gem"]].copy()
    labels["appid"] = pd.to_numeric(labels["appid"], errors="coerce").astype("Int64")
    labels = labels.dropna(subset=["appid"]).copy()
    labels["appid"] = labels["appid"].astype(int)
    labels["is_gem"] = labels["is_gem"].astype(bool).astype(int)

    # Per-game sentiment aggregates from reviews.
    if "gameId" not in reviews.columns:
        raise ValueError("reviews_csv must have a 'gameId' column")
    sentiment_per_game = aggregate_sentiment_per_game(reviews, game_id_col="gameId", text_col="review")

    games = _add_release_year(games)
    if include_temporal_split_key:
        games = _add_temporal_split_key(games)
    _coerce_bool_int(games, ["is_free", "windows", "mac", "linux", "coming_soon"])

    df = games.merge(sentiment_per_game, left_on="id", right_on="gameId", how="left")
    if "gameId" in df.columns:
        df = df.drop(columns=["gameId"])

    df = df.merge(labels, left_on="id", right_on="appid", how="left")
    if "appid" in df.columns:
        df = df.drop(columns=["appid"])

    if require_labels:
        df = df.dropna(subset=["is_gem"]).copy()
    _validate_label_distribution(df, min_positive=min_positive, min_positive_ratio=min_positive_ratio)

    if require_labels:
        df["is_gem"] = df["is_gem"].astype(int)
        df["has_label"] = 1
    else:
        df["has_label"] = df["is_gem"].notna().astype(int)
        df["is_gem"] = df["is_gem"].astype("Int64")

    df, genre_cols = _add_multihot_features(
        df,
        source_col="genre_names",
        prefix="genre",
        top_k=max(int(genre_top_k), 0),
    )
    df, category_cols = _add_multihot_features(
        df,
        source_col="category_names",
        prefix="category",
        top_k=max(int(category_top_k), 0),
    )
    interaction_cols: list[str] = []
    if include_interactions:
        df, interaction_cols = _add_interaction_features(df)

    keep_cols = [c for c in (GAME_BASE_FEATURE_COLS + ["release_year"]) if c in df.columns]
    if include_temporal_split_key and "temporal_split_key" in df.columns:
        keep_cols += ["temporal_split_key"]
    keep_cols += [c for c in df.columns if c.startswith("sentiment_") or c in {"avg_review_length", "review_count"}]
    keep_cols += genre_cols + category_cols + interaction_cols
    keep_cols += ["has_label"]
    keep_cols += ["is_gem"]
    df = df[keep_cols].copy()

    # Ensure all numeric-ish columns are non-null.
    feature_cols = [c for c in df.columns if c not in {"is_gem", "temporal_split_key"}]
    df[feature_cols] = df[feature_cols].fillna(0)
    if require_labels:
        df["is_gem"] = df["is_gem"].fillna(0).astype(int)

    summary = {
        "n_games_rows": int(len(games)),
        "n_reviews_rows": int(len(reviews)),
        "n_labels_rows": int(len(labels)),
        "n_output_rows": int(len(df)),
        "n_labeled_rows": int(df["has_label"].sum()),
        "n_positive": int((df["is_gem"] == 1).sum()),
        "n_negative": int((df["is_gem"] == 0).sum()),
        "n_genre_features": int(len(genre_cols)),
        "n_category_features": int(len(category_cols)),
        "n_interaction_features": int(len(interaction_cols)),
        "n_feature_columns": int(len([c for c in df.columns if c not in {"is_gem"}])),
    }
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a supervised training CSV by joining cleaned Steam data with curated labels."
    )
    ap.add_argument("--games-csv", default=local_data_file("steam_games_clean.csv"), help="Input games CSV")
    ap.add_argument("--reviews-csv", default=local_data_file("steam_reviews_clean.csv"), help="Input reviews CSV")
    ap.add_argument("--labels-csv", default=local_data_file("curated_steam_labels.csv"), help="Curated labels export (appid,is_gem)")
    ap.add_argument("--out-csv", default=local_data_file("training_dataset.csv"), help="Output training CSV")
    ap.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep rows without labels (is_gem will remain empty). Default drops unlabeled rows.",
    )
    ap.add_argument(
        "--include-temporal-split-key",
        action="store_true",
        help="Add temporal_split_key derived from release_date for time-aware holdout experiments.",
    )
    ap.add_argument(
        "--min-positive",
        type=int,
        default=10,
        help="Minimum number of positive labels required for supervised dataset creation.",
    )
    ap.add_argument(
        "--min-positive-ratio",
        type=float,
        default=0.05,
        help="Minimum positive-label ratio required for supervised dataset creation.",
    )
    ap.add_argument(
        "--summary-out",
        default=model_artifact_file("training_dataset_summary.json"),
        help="Optional output path for dataset summary JSON.",
    )
    ap.add_argument(
        "--genre-top-k",
        type=int,
        default=40,
        help="Number of most frequent genres to expand into multi-hot features.",
    )
    ap.add_argument(
        "--category-top-k",
        type=int,
        default=40,
        help="Number of most frequent categories to expand into multi-hot features.",
    )
    ap.add_argument(
        "--no-interactions",
        action="store_true",
        help="Disable curated numeric interaction feature generation.",
    )
    args = ap.parse_args()

    out_df, summary = build_training_dataset(
        games_csv=Path(args.games_csv),
        reviews_csv=Path(args.reviews_csv),
        labels_csv=Path(args.labels_csv),
        require_labels=not args.keep_unlabeled,
        include_temporal_split_key=args.include_temporal_split_key,
        min_positive=args.min_positive,
        min_positive_ratio=args.min_positive_ratio,
        genre_top_k=args.genre_top_k,
        category_top_k=args.category_top_k,
        include_interactions=not args.no_interactions,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("Built training dataset.")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"out_csv: {out_path}")
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary_out: {summary_out}")


if __name__ == "__main__":
    main()

