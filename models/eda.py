"""Exploratory Data Analysis for Steam game data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for all numeric columns."""
    return df.select_dtypes(include=[np.number]).describe().T


def missing_data_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values per column, sorted by severity."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2) if total else missing * 0
    return (
        pd.DataFrame({"missing_count": missing, "missing_pct": pct, "dtype": df.dtypes})
        .sort_values("missing_pct", ascending=False)
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between numeric columns."""
    return df.select_dtypes(include=[np.number]).corr()


def review_score_distribution(games_df: pd.DataFrame) -> pd.Series:
    """Value counts of Steam review score descriptions."""
    if "reviewScoreDesc" in games_df.columns:
        return games_df["reviewScoreDesc"].value_counts()
    return pd.Series(dtype=int)


def platform_distribution(games_df: pd.DataFrame) -> pd.DataFrame:
    """Count of games supporting each platform."""
    platforms = [p for p in ("windows", "mac", "linux") if p in games_df.columns]
    if not platforms:
        return pd.DataFrame()
    return games_df[platforms].sum().to_frame("count")


def ownership_tiers(games_df: pd.DataFrame) -> pd.Series:
    """Bin games into ownership tiers based on owners_max."""
    if "owners_max" not in games_df.columns:
        return pd.Series(dtype=str)
    bins = [0, 20_000, 200_000, 1_000_000, 10_000_000, float("inf")]
    labels = ["<20k", "20k-200k", "200k-1M", "1M-10M", "10M+"]
    return pd.cut(games_df["owners_max"], bins=bins, labels=labels).value_counts().sort_index()
