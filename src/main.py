from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Allow running as either:
# - `python -m src.main`
# - `python src/main.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig, load_config
from src.io_utils import write_json
from src.pipeline import run_preprocessing_pipeline

PROFILE_CHOICES = ("full", "quick", "no-llm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent-based social survey preprocessing pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="full",
        help="Run profile: full, quick, or no-llm.",
    )
    return parser


def _config_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _apply_profile(config: AppConfig, profile: str) -> AppConfig:
    if profile == "full":
        return config
    if profile == "quick":
        return replace(
            config,
            embeddings=replace(config.embeddings, enabled=False),
            clustering=replace(config.clustering, enabled=False),
            cluster_interpretation=replace(config.cluster_interpretation, enabled=False),
            merge_analysis=replace(config.merge_analysis, enabled=False),
            final_report=replace(config.final_report, enabled=False),
            gigachat=replace(config.gigachat, enabled=False),
        )
    # profile == no-llm
    return replace(
        config,
        gigachat=replace(config.gigachat, enabled=False),
    )


def _materialize_quick_sample(config: AppConfig, sample_rows: int = 80) -> AppConfig:
    src = config.paths.input_dataset
    suffix = src.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(src, nrows=sample_rows)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(src).head(sample_rows)
    else:
        return config

    sample_path = config.paths.data_raw_dir / "quick_sample.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sample_path, index=False)
    return replace(config, paths=replace(config.paths, input_dataset=sample_path))


def _write_run_metadata(config: AppConfig, payload: dict) -> None:
    log_path = config.paths.artifacts_logs_dir / "run_metadata.json"
    write_json(log_path, payload)


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    started = time.time()
    config = load_config(config_path)
    config = _apply_profile(config, args.profile)
    if args.profile == "quick":
        config = _materialize_quick_sample(config)

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "config_path": str(config_path),
        "config_hash_sha256": _config_hash(config_path),
        "success": False,
        "runtime_seconds": None,
        "error": None,
        "summary": None,
    }

    try:
        summary = run_preprocessing_pipeline(config)
        run_info["success"] = True
        run_info["summary"] = summary.as_dict()
        print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    except Exception as exc:
        run_info["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        run_info["runtime_seconds"] = round(time.time() - started, 3)
        _write_run_metadata(config, run_info)


if __name__ == "__main__":
    main()
