from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LOCAL = PROJECT_ROOT / "data" / "local"
ARTIFACTS = PROJECT_ROOT / "artifacts"
MODELS_ARTIFACTS = ARTIFACTS / "models"
ANALYSIS_CACHE = ARTIFACTS / "analysis"


def bootstrap_project_root() -> Path:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def local_data_file(name: str) -> str:
    return str(DATA_LOCAL / name)


def model_artifact_file(name: str) -> str:
    return str(MODELS_ARTIFACTS / name)

