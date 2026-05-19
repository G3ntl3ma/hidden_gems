import csv
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Set

import requests

from scripts._runtime import DATA_LOCAL, bootstrap_project_root


STEAM_GET_APP_LIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"


class SteamApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class SteamStoreClient:
    api_key: str
    timeout_s: float = 30.0

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        merged_params = {
            "key": self.api_key,
            "max_results": 50000,
            "include_games": True,
            "include_dlc": False,
            "include_software": False,
            "include_videos": False,
            "include_hardware": False,
            **params,
        }
        try:
            resp = requests.get(
                STEAM_GET_APP_LIST_URL,
                params=merged_params,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise SteamApiError(f"Failed to call Steam GetAppList: {exc}") from exc

        if resp.status_code != 200:
            raise SteamApiError(
                f"Steam GetAppList returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise SteamApiError("Steam GetAppList returned invalid JSON") from exc

        if not isinstance(data, dict) or "response" not in data:
            raise SteamApiError("Steam GetAppList payload missing 'response' field")

        return data["response"]

    def iter_all_appids(self) -> Iterable[int]:
        """
        Yield all appids by paginating over GetAppList using last_appid.
        """
        last_appid: int | None = None

        while True:
            params: dict[str, Any] = {}
            if last_appid is not None:
                params["last_appid"] = last_appid

            payload = self._request(params)
            apps = payload.get("apps") or []

            if not isinstance(apps, list):
                raise SteamApiError("Steam GetAppList 'apps' field is not a list")

            if not apps:
                break

            for app in apps:
                # Typical shape: {"appid": 10, "name": "Counter-Strike"}
                appid = app.get("appid")
                if isinstance(appid, int):
                    yield appid

            # Prepare for next page using the last appid we successfully read
            last_appid = apps[-1].get("appid")
            if not isinstance(last_appid, int):
                # If the last entry has no valid appid, stop to avoid infinite loop
                break


def collect_unique_appids(client: SteamStoreClient) -> list[int]:
    seen: Set[int] = set()
    for appid in client.iter_all_appids():
        seen.add(appid)
    return sorted(seen)


def write_appids_csv(appids: Iterable[int], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["appid"])
        for appid in appids:
            writer.writerow([appid])


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"Environment variable {name} is not set. "
            "Set it in your shell or .env before running this script."
        )
    return value


def main() -> None:
    bootstrap_project_root()
    parser = argparse.ArgumentParser(description="Export Steam app IDs to CSV.")
    parser.add_argument(
        "--out",
        default=str(DATA_LOCAL / "steam_appids.csv"),
        help="Output CSV path (default: data/local/steam_appids.csv).",
    )
    args = parser.parse_args()

    api_key = _get_required_env("STEAM_WEB_API_KEY")
    output_path = args.out

    client = SteamStoreClient(api_key=api_key)
    appids = collect_unique_appids(client)
    write_appids_csv(appids, output_path)
    print(f"Wrote {len(appids)} appids to {output_path}")


if __name__ == "__main__":
    main()

