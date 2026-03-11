from __future__ import annotations

import argparse

from steam_scraper.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Steam data ingestion pipeline")
    parser.add_argument(
        "--appid-csv",
        dest="appid_csv",
        default=None,
        help="Path to steam_appids.csv (default: steam_appids.csv in project root)",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        default=None,
        help="Path to output CSV file (default: steam_games_full.csv in project root)",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Limit the number of appids to process (useful for testing)",
    )
    parser.add_argument(
        "--offset",
        dest="offset",
        type=int,
        default=0,
        help="Skip the first N appids (useful for resuming)",
    )
    args = parser.parse_args()

    run_pipeline(
        appids_csv=args.appid_csv,
        output_csv=args.output_csv,
        limit=args.limit,
        offset=args.offset,
    )


if __name__ == "__main__":
    main()

