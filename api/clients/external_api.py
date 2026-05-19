from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .json_http import JsonHttpClient


@dataclass(frozen=True)
class ExternalApiClient:
    base_url: str
    timeout_s: float = 10.0

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        return JsonHttpClient(timeout_s=self.timeout_s).get_json(url, params=params)

