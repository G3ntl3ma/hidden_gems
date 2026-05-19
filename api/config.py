from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env", override=False)
    database_url = os.getenv("DATABASE_URL", f"file:{repo_root / 'dev.db'}")
    return Settings(database_url=database_url)

