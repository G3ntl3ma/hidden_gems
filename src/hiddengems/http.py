from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class JsonHttpClient:
    timeout_s: float = 10.0

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        response = requests.get(url, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

