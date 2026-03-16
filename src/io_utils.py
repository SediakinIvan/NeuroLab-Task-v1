from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import AppConfig


def ensure_project_dirs(config: AppConfig) -> None:
    dirs = [
        config.paths.data_raw_dir,
        config.paths.data_interim_dir,
        config.paths.data_processed_dir,
        config.paths.artifacts_reports_dir,
        config.paths.artifacts_plots_dir,
        config.paths.artifacts_models_dir,
        config.paths.artifacts_logs_dir,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
