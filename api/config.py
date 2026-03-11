from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str


def get_settings() -> Settings:
    load_dotenv(override=False)
    database_url = os.getenv("DATABASE_URL", "file:./dev.db")
    return Settings(database_url=database_url)

