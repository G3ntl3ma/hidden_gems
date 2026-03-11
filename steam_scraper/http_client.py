from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import get_api_config


@dataclass
class HttpClient:
    timeout_s: float = get_api_config().http_timeout_s

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        resp = requests.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()


class SteamSpyClient:
    """
    Thin wrapper around SteamSpy with basic rate limiting.
    """

    def __init__(self) -> None:
        self._cfg = get_api_config()
        self._http = HttpClient(timeout_s=self._cfg.http_timeout_s)
        self._last_request_ts: float | None = None

    def _throttle(self) -> None:
        if self._last_request_ts is None:
            return
        elapsed = time.time() - self._last_request_ts
        if elapsed < self._cfg.steamspy_min_interval_s:
            time.sleep(self._cfg.steamspy_min_interval_s - elapsed)

    def get_appdetails(self, appid: int) -> dict[str, Any] | None:
        """
        Returns SteamSpy appdetails JSON for a single app or None on error.
        """
        from .config import get_api_config

        cfg = get_api_config()
        base = cfg.steamspy_base_url.rstrip("/")
        url = f"{base}/api.php"

        self._throttle()
        try:
            data = self._http.get_json(
                url,
                params={"request": "appdetails", "appid": str(appid)},
            )
        except requests.RequestException:
            return None
        finally:
            self._last_request_ts = time.time()

        if not isinstance(data, dict):
            return None
        return data

