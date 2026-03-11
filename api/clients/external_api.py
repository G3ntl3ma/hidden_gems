from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ExternalApiClient:
    base_url: str
    timeout_s: float = 10.0

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

