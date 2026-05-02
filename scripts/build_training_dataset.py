from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Make project root importable when executing this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def build_training_dataset(
    *,
    games_csv: Path,
    reviews_csv: Path,
    labels_csv: Path,
    require_labels: bool,
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
    _coerce_bool_int(games, ["is_free", "windows", "mac", "linux", "coming_soon"])

    df = games.merge(sentiment_per_game, left_on="id", right_on="gameId", how="left")
    if "gameId" in df.columns:
        df = df.drop(columns=["gameId"])

    df = df.merge(labels, left_on="id", right_on="appid", how="left")
    if "appid" in df.columns:
        df = df.drop(columns=["appid"])

    if require_labels:
        df = df.dropna(subset=["is_gem"]).copy()

    df["is_gem"] = df["is_gem"].fillna(0).astype(int)

    keep_cols = [c for c in (GAME_BASE_FEATURE_COLS + ["release_year"]) if c in df.columns]
    keep_cols += [c for c in df.columns if c.startswith("sentiment_") or c in {"avg_review_length", "review_count"}]
    keep_cols += ["is_gem"]
    df = df[keep_cols].copy()

    # Ensure all numeric-ish columns are non-null.
    feature_cols = [c for c in df.columns if c != "is_gem"]
    df[feature_cols] = df[feature_cols].fillna(0)

    summary = {
        "n_games_rows": int(len(games)),
        "n_reviews_rows": int(len(reviews)),
        "n_labels_rows": int(len(labels)),
        "n_output_rows": int(len(df)),
        "n_positive": int(df["is_gem"].sum()),
        "n_negative": int((df["is_gem"] == 0).sum()),
    }
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a supervised training CSV by joining cleaned Steam data with curated labels."
    )
    ap.add_argument("--games-csv", default="steam_games_clean.csv", help="Input games CSV")
    ap.add_argument("--reviews-csv", default="steam_reviews_clean.csv", help="Input reviews CSV")
    ap.add_argument("--labels-csv", default="curated_steam_labels.csv", help="Curated labels export (appid,is_gem)")
    ap.add_argument("--out-csv", default="training_dataset.csv", help="Output training CSV")
    ap.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep rows without labels (is_gem will be 0). Default drops unlabeled rows.",
    )
    args = ap.parse_args()

    out_df, summary = build_training_dataset(
        games_csv=Path(args.games_csv),
        reviews_csv=Path(args.reviews_csv),
        labels_csv=Path(args.labels_csv),
        require_labels=not args.keep_unlabeled,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("Built training dataset.")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"out_csv: {out_path}")


if __name__ == "__main__":
    main()

