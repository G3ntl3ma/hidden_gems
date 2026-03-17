from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.analysis_artifacts import default_cache_dir, load_or_precompute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute and persist analysis artifacts for the Streamlit dashboard."
    )
    parser.add_argument(
        "--games-in",
        default="steam_games_clean.csv",
        help="Path to cleaned games CSV input (default: steam_games_clean.csv).",
    )
    parser.add_argument(
        "--reviews-in",
        default="steam_reviews_clean.csv",
        help="Path to cleaned reviews CSV input (default: steam_reviews_clean.csv).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(default_cache_dir()),
        help="Directory to store analysis artifacts (default: .cache/analysis).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if manifest is still fresh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games_csv = Path(args.games_in)
    reviews_csv = Path(args.reviews_in)
    cache_dir = Path(args.cache_dir)

    if not games_csv.exists():
        raise FileNotFoundError(f"Games CSV does not exist: {games_csv}")
    if not reviews_csv.exists():
        raise FileNotFoundError(f"Reviews CSV does not exist: {reviews_csv}")

    _, from_cache, resolved_cache_dir = load_or_precompute(
        games_csv=games_csv,
        reviews_csv=reviews_csv,
        cache_dir=cache_dir,
        force_recompute=args.force,
    )

    if from_cache:
        print("Analysis artifacts are up to date; loaded from cache.")
    else:
        print("Analysis artifacts recomputed and saved.")
    print(f"Cache directory: {resolved_cache_dir}")
    print(f"Manifest: {resolved_cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
