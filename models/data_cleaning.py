from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
import re
import unicodedata
from typing import Any

import pandas as pd


CONTROL_CHAR_RE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
WORD_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class CleaningResult:
    games: pd.DataFrame
    reviews: pd.DataFrame
    report: dict[str, Any]


def normalize_text_conservative(value: Any) -> str:
    """
    Apply conservative cleanup suitable for multilingual NLP:
    - normalize unicode to NFKC
    - unescape HTML entities
    - strip HTML tags and control chars
    - collapse repeated whitespace
    """
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def has_lexical_content(value: Any) -> bool:
    text = normalize_text_conservative(value)
    return bool(WORD_TOKEN_RE.search(text))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _coerce_numeric_non_negative(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    changes: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            continue
        original = df[col]
        numeric = pd.to_numeric(original, errors="coerce")
        negative_mask = numeric < 0
        coerced_mask = numeric.isna() & original.notna()
        numeric = numeric.fillna(0)
        numeric[negative_mask] = 0
        df[col] = numeric.astype("int64")
        changes[f"{col}_coerced_or_clamped"] = int((negative_mask | coerced_mask).sum())
    return changes


def _coerce_datetime_iso(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    changes: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            continue
        original = df[col]
        parsed = pd.to_datetime(original, errors="coerce", utc=False)
        invalid_mask = original.notna() & original.astype(str).str.strip().ne("") & parsed.isna()
        df[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S").where(parsed.notna(), "")
        changes[f"{col}_invalid"] = int(invalid_mask.sum())
    return changes


def clean_games_dataframe(games_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = games_df.copy()
    report: dict[str, Any] = {"input_rows": int(len(df))}

    # Normalize text columns conservatively.
    text_columns = [
        "name",
        "detailed_description",
        "about_the_game",
        "short_description",
        "supported_languages",
        "developers",
        "publishers",
        "genre_names",
        "category_names",
        "header_image",
        "reviewScoreDesc",
        "store_status",
        "steamspy_status",
        "reviews_status",
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].map(normalize_text_conservative)

    # ID coercion and invalid-row removal.
    if "id" not in df.columns:
        raise ValueError("Games CSV must include an 'id' column")
    game_ids = pd.to_numeric(df["id"], errors="coerce")
    invalid_id_mask = game_ids.isna()
    report["rows_with_invalid_id"] = int(invalid_id_mask.sum())
    df = df[~invalid_id_mask].copy()
    df["id"] = game_ids[~invalid_id_mask].astype("int64")

    # Drop structurally empty game rows.
    key_text_columns = [
        "name",
        "short_description",
        "about_the_game",
        "detailed_description",
    ]
    available_keys = [c for c in key_text_columns if c in df.columns]
    if available_keys:
        empty_key_mask = df[available_keys].apply(
            lambda row: all(not str(v).strip() for v in row), axis=1
        )
    else:
        empty_key_mask = pd.Series(False, index=df.index)
    report["rows_with_empty_key_text"] = int(empty_key_mask.sum())
    df = df[~empty_key_mask].copy()

    # Numeric consistency.
    report.update(
        _coerce_numeric_non_negative(
            df,
            [
                "required_age",
                "metacritic",
                "owners_min",
                "owners_max",
                "average_2weeks",
                "average_forever",
                "median_forever",
                "median_2weeks",
                "ccu",
                "reviewNumReviews",
                "reviewTotalPositive",
                "reviewTotalNegative",
                "reviewTotalReviews",
            ],
        )
    )

    if "reviewScore" in df.columns:
        original = df["reviewScore"]
        score = pd.to_numeric(original, errors="coerce")
        invalid = score.isna() & original.notna()
        score = score.clip(lower=0, upper=10)
        df["reviewScore"] = score.fillna(0).astype("int64")
        report["reviewScore_coerced_or_clamped"] = int((invalid | (score != pd.to_numeric(original, errors="coerce"))).fillna(False).sum())

    if {"owners_min", "owners_max"}.issubset(df.columns):
        inverted = df["owners_max"] < df["owners_min"]
        report["owners_range_inverted_rows"] = int(inverted.sum())
        df.loc[inverted, "owners_max"] = df.loc[inverted, "owners_min"]

    # Boolean coercion.
    for col in ["is_free", "windows", "mac", "linux", "coming_soon"]:
        if col in df.columns:
            df[col] = df[col].map(_coerce_bool)

    # release_date normalization (date only).
    if "release_date" in df.columns:
        parsed_release = pd.to_datetime(df["release_date"], errors="coerce", utc=False)
        invalid_release = (
            df["release_date"].notna()
            & df["release_date"].astype(str).str.strip().ne("")
            & parsed_release.isna()
        )
        df["release_date"] = parsed_release.dt.strftime("%Y-%m-%d").where(parsed_release.notna(), "")
        report["release_date_invalid"] = int(invalid_release.sum())

    # Deduplicate games by id, keeping the most recently updated row when available.
    duplicate_mask = df.duplicated(subset=["id"], keep=False)
    report["duplicate_game_ids"] = int(duplicate_mask.sum())
    if "last_updated" in df.columns:
        parsed_last_updated = pd.to_datetime(df["last_updated"], errors="coerce", utc=False)
        df["_sort_last_updated"] = parsed_last_updated
        df = df.sort_values(by=["id", "_sort_last_updated"], ascending=[True, False], na_position="last")
        df = df.drop(columns=["_sort_last_updated"])
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    report["output_rows"] = int(len(df))
    report["rows_dropped_total"] = report["input_rows"] - report["output_rows"]
    return df, report


def clean_reviews_dataframe(reviews_df: pd.DataFrame, valid_game_ids: set[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = reviews_df.copy()
    report: dict[str, Any] = {"input_rows": int(len(df))}

    if "id" not in df.columns:
        raise ValueError("Reviews CSV must include an 'id' column")
    if "gameId" not in df.columns:
        raise ValueError("Reviews CSV must include a 'gameId' column")

    # Normalize text fields.
    for col in ["id", "authorSteamId", "language", "review"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_text_conservative)

    # Remove invalid IDs.
    invalid_review_id_mask = df["id"].astype(str).str.strip().eq("")
    report["rows_with_invalid_review_id"] = int(invalid_review_id_mask.sum())
    df = df[~invalid_review_id_mask].copy()

    # Normalize and validate game IDs.
    game_ids = pd.to_numeric(df["gameId"], errors="coerce")
    invalid_game_id_mask = game_ids.isna()
    report["rows_with_invalid_game_id"] = int(invalid_game_id_mask.sum())
    df = df[~invalid_game_id_mask].copy()
    df["gameId"] = game_ids[~invalid_game_id_mask].astype("int64")

    unknown_game_mask = ~df["gameId"].isin(valid_game_ids)
    report["rows_with_unknown_game_id"] = int(unknown_game_mask.sum())
    df = df[~unknown_game_mask].copy()

    # Timestamp parsing for deterministic dedupe ordering.
    parsed_updated = pd.to_datetime(df.get("timestampUpdated", pd.Series("", index=df.index)), errors="coerce", utc=False)
    parsed_created = pd.to_datetime(df.get("timestampCreated", pd.Series("", index=df.index)), errors="coerce", utc=False)
    if "timestampUpdated" in df.columns:
        invalid_updated = (
            df["timestampUpdated"].notna()
            & df["timestampUpdated"].astype(str).str.strip().ne("")
            & parsed_updated.isna()
        )
        report["timestampUpdated_invalid"] = int(invalid_updated.sum())
    if "timestampCreated" in df.columns:
        invalid_created = (
            df["timestampCreated"].notna()
            & df["timestampCreated"].astype(str).str.strip().ne("")
            & parsed_created.isna()
        )
        report["timestampCreated_invalid"] = int(invalid_created.sum())

    # Deduplicate reviews by review ID, keeping latest updated/created row.
    duplicate_ids = df.duplicated(subset=["id"], keep=False)
    report["duplicate_review_ids"] = int(duplicate_ids.sum())
    df["_sort_updated"] = parsed_updated
    df["_sort_created"] = parsed_created
    df = df.sort_values(
        by=["id", "_sort_updated", "_sort_created"],
        ascending=[True, False, False],
        na_position="last",
    )
    df = df.drop_duplicates(subset=["id"], keep="first")
    df = df.drop(columns=["_sort_updated", "_sort_created"])

    # Remove tokenless reviews.
    tokenless_mask = ~df["review"].map(has_lexical_content)
    report["rows_with_tokenless_review"] = int(tokenless_mask.sum())
    df = df[~tokenless_mask].copy()

    # Numeric and boolean cleanup.
    report.update(
        _coerce_numeric_non_negative(
            df,
            [
                "authorPlaytimeForever",
                "authorPlaytimeAtReview",
                "votesUp",
                "votesFunny",
            ],
        )
    )
    if "weightedVoteScore" in df.columns:
        score = pd.to_numeric(df["weightedVoteScore"], errors="coerce")
        invalid_score = score.isna() & df["weightedVoteScore"].notna()
        df["weightedVoteScore"] = score.fillna(0.0).astype(float)
        report["weightedVoteScore_invalid"] = int(invalid_score.sum())

    for col in ["votedUp", "writtenDuringEarlyAccess"]:
        if col in df.columns:
            df[col] = df[col].map(_coerce_bool)

    # Datetime normalization.
    report.update(
        _coerce_datetime_iso(
            df,
            ["authorLastPlayed", "timestampCreated", "timestampUpdated"],
        )
    )

    report["output_rows"] = int(len(df))
    report["rows_dropped_total"] = report["input_rows"] - report["output_rows"]
    return df.reset_index(drop=True), report


def _quality_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    games = report["games"]
    reviews = report["reviews"]

    if games["input_rows"] > 0 and games["rows_dropped_total"] / games["input_rows"] > 0.05:
        warnings.append("More than 5% of game rows were dropped during cleaning.")
    if reviews["input_rows"] > 0 and reviews["rows_dropped_total"] / reviews["input_rows"] > 0.10:
        warnings.append("More than 10% of review rows were dropped during cleaning.")
    if reviews.get("duplicate_review_ids", 0) > 0:
        warnings.append("Duplicate review IDs were found and deduplicated.")
    if reviews.get("rows_with_unknown_game_id", 0) > 0:
        warnings.append("Some reviews referenced game IDs not present in cleaned games.")
    return warnings


def clean_datasets(games_df: pd.DataFrame, reviews_df: pd.DataFrame) -> CleaningResult:
    cleaned_games, games_report = clean_games_dataframe(games_df)
    valid_game_ids = set(cleaned_games["id"].astype(int).tolist())
    cleaned_reviews, reviews_report = clean_reviews_dataframe(reviews_df, valid_game_ids)

    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "games": games_report,
        "reviews": reviews_report,
        "warnings": [],
    }
    report["warnings"] = _quality_warnings(report)
    return CleaningResult(games=cleaned_games, reviews=cleaned_reviews, report=report)


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
