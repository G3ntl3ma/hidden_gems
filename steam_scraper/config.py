from __future__ import annotations

from dataclasses import dataclass


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
    appids_csv: str = "steam_appids.csv"
    output_csv: str = "steam_games_full.csv"
    reviews_csv: str = "steam_reviews_full.csv"


def get_api_config() -> ApiConfig:
    return ApiConfig()


def get_paths_config() -> PathsConfig:
    return PathsConfig()

