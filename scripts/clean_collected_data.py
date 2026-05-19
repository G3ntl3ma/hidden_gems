from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._runtime import bootstrap_project_root, local_data_file

bootstrap_project_root()

from models.data_cleaning import clean_datasets, report_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean collected Steam game/review datasets and generate a quality report."
    )
    parser.add_argument(
        "--games-in",
        default=local_data_file("steam_games_full.csv"),
        help="Path to raw games CSV (default: data/local/steam_games_full.csv).",
    )
    parser.add_argument(
        "--reviews-in",
        default=local_data_file("steam_reviews_full.csv"),
        help="Path to raw reviews CSV (default: data/local/steam_reviews_full.csv).",
    )
    parser.add_argument(
        "--games-out",
        default=local_data_file("steam_games_clean.csv"),
        help="Path to cleaned games CSV output (default: data/local/steam_games_clean.csv).",
    )
    parser.add_argument(
        "--reviews-out",
        default=local_data_file("steam_reviews_clean.csv"),
        help="Path to cleaned reviews CSV output (default: data/local/steam_reviews_clean.csv).",
    )
    parser.add_argument(
        "--report-out",
        default=local_data_file("data_quality_report.json"),
        help="Path to JSON quality report output (default: data/local/data_quality_report.json).",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return pd.read_csv(path)


def _print_summary(report: dict[str, object]) -> None:
    games = report["games"]
    reviews = report["reviews"]
    warnings = report.get("warnings", [])

    print("Cleaning complete.")
    print(
        f"Games: {games['input_rows']} -> {games['output_rows']} "
        f"(dropped {games['rows_dropped_total']})"
    )
    print(
        f"Reviews: {reviews['input_rows']} -> {reviews['output_rows']} "
        f"(dropped {reviews['rows_dropped_total']})"
    )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def main() -> None:
    args = parse_args()
    games_in = Path(args.games_in)
    reviews_in = Path(args.reviews_in)
    games_out = Path(args.games_out)
    reviews_out = Path(args.reviews_out)
    report_out = Path(args.report_out)

    games_df = _read_csv(games_in)
    reviews_df = _read_csv(reviews_in)

    result = clean_datasets(games_df, reviews_df)

    result.games.to_csv(games_out, index=False)
    result.reviews.to_csv(reviews_out, index=False)
    report_out.write_text(report_to_json(result.report) + "\n", encoding="utf-8")

    _print_summary(result.report)
    print(f"Cleaned games CSV: {games_out}")
    print(f"Cleaned reviews CSV: {reviews_out}")
    print(f"Quality report: {report_out}")


if __name__ == "__main__":
    main()
