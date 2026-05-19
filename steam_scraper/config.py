from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_LOCAL = _PROJECT_ROOT / "data" / "local"


@dataclass(frozen=True)
class ApiConfig:
    store_base_url: str = "https://store.steampowered.com"
    steamspy_base_url: str = "https://steamspy.com"

    # Timeouts (seconds)
    http_timeout_s: float = 15.0

    # Rate limiting
    # SteamSpy docs recommend 1 request per second for most endpoints.
    steamspy_min_interval_s: float = 1.1


@dataclass(frozen=True)
class PathsConfig:
    appids_csv: str = str(_DATA_LOCAL / "steam_appids.csv")
    output_csv: str = str(_DATA_LOCAL / "steam_games_full.csv")
    reviews_csv: str = str(_DATA_LOCAL / "steam_reviews_full.csv")


def get_api_config() -> ApiConfig:
    return ApiConfig()


def get_paths_config() -> PathsConfig:
    return PathsConfig()

