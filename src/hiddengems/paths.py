from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runtime import project_root


@dataclass(frozen=True)
class DataLayout:
    root: Path
    data_dir: Path
    reference_dir: Path
    local_dir: Path
    artifacts_dir: Path
    analysis_cache_dir: Path
    models_artifacts_dir: Path
    steam_appids_csv: Path
    steam_games_full_csv: Path
    steam_reviews_full_csv: Path
    steam_games_clean_csv: Path
    steam_reviews_clean_csv: Path
    curated_labels_csv: Path
    training_dataset_csv: Path
    training_dataset_eval_csv: Path
    data_quality_report_json: Path


def get_data_layout() -> DataLayout:
    root = project_root()
    data_dir = root / "data"
    local_dir = data_dir / "local"
    artifacts_dir = root / "artifacts"
    return DataLayout(
        root=root,
        data_dir=data_dir,
        reference_dir=data_dir / "reference",
        local_dir=local_dir,
        artifacts_dir=artifacts_dir,
        analysis_cache_dir=artifacts_dir / "analysis",
        models_artifacts_dir=artifacts_dir / "models",
        steam_appids_csv=local_dir / "steam_appids.csv",
        steam_games_full_csv=local_dir / "steam_games_full.csv",
        steam_reviews_full_csv=local_dir / "steam_reviews_full.csv",
        steam_games_clean_csv=local_dir / "steam_games_clean.csv",
        steam_reviews_clean_csv=local_dir / "steam_reviews_clean.csv",
        curated_labels_csv=local_dir / "curated_steam_labels.csv",
        training_dataset_csv=local_dir / "training_dataset.csv",
        training_dataset_eval_csv=local_dir / "training_dataset_eval.csv",
        data_quality_report_json=local_dir / "data_quality_report.json",
    )

