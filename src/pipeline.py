from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import AppConfig
import numpy as np

from src.data.loader import load_dataset
from src.data.splitter import split_dataset
from src.data.validation import build_validation_report, ensure_stable_id, infer_column_types
from src.eda.analyzer import run_eda
from src.io_utils import ensure_project_dirs, write_json
from src.llm.cluster_interpreter import run_cluster_interpretation
from src.llm.final_report import generate_final_report
from src.merge.analyzer import merge_and_analyze
from src.text.clustering import run_clustering
from src.text.embeddings import build_embeddings
from src.text.preprocess import preprocess_text_dataframe


@dataclass(frozen=True)
class RunSummary:
    input_rows: int
    input_columns: int
    id_column: str
    structured_path: Path
    text_path: Path
    validation_report_path: Path
    column_types_path: Path
    eda_summary_path: Path
    outlier_flags_path: Path
    text_preprocessing_path: Path | None
    embedding_meta_path: Path | None
    embeddings_path: Path | None
    text_clusters_path: Path | None
    cluster_interpretations_path: Path | None
    final_dataset_path: Path | None
    cluster_correlations_path: Path | None
    final_report_json_path: Path | None
    final_report_markdown_path: Path | None
    final_report_context_path: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "input_columns": self.input_columns,
            "id_column": self.id_column,
            "structured_path": str(self.structured_path),
            "text_path": str(self.text_path),
            "validation_report_path": str(self.validation_report_path),
            "column_types_path": str(self.column_types_path),
            "eda_summary_path": str(self.eda_summary_path),
            "outlier_flags_path": str(self.outlier_flags_path),
            "text_preprocessing_path": str(self.text_preprocessing_path) if self.text_preprocessing_path else None,
            "embedding_meta_path": str(self.embedding_meta_path) if self.embedding_meta_path else None,
            "embeddings_path": str(self.embeddings_path) if self.embeddings_path else None,
            "text_clusters_path": str(self.text_clusters_path) if self.text_clusters_path else None,
            "cluster_interpretations_path": (
                str(self.cluster_interpretations_path) if self.cluster_interpretations_path else None
            ),
            "final_dataset_path": str(self.final_dataset_path) if self.final_dataset_path else None,
            "cluster_correlations_path": (
                str(self.cluster_correlations_path) if self.cluster_correlations_path else None
            ),
            "final_report_json_path": str(self.final_report_json_path) if self.final_report_json_path else None,
            "final_report_markdown_path": (
                str(self.final_report_markdown_path) if self.final_report_markdown_path else None
            ),
            "final_report_context_path": (
                str(self.final_report_context_path) if self.final_report_context_path else None
            ),
        }


def run_preprocessing_pipeline(config: AppConfig) -> RunSummary:
    ensure_project_dirs(config)

    raw_df = load_dataset(config.paths.input_dataset)
    df, id_column = ensure_stable_id(raw_df, config.columns.id_column)
    column_types = infer_column_types(df, config.columns, id_column=id_column)

    validation_report = build_validation_report(df, id_column=id_column, column_types=column_types)
    split = split_dataset(df, column_types=column_types, split_cfg=config.split)

    structured_path = config.paths.data_interim_dir / "structured.parquet"
    text_path = config.paths.data_interim_dir / "text.parquet"
    text_df = split.text_df
    text_preprocessing_path: Path | None = None
    embedding_source_column = config.text_processing.source_column
    if config.text_processing.enabled:
        preprocess = preprocess_text_dataframe(text_df, config.text_processing)
        text_df = preprocess.text_df
        text_preprocessing_path = config.paths.artifacts_reports_dir / "text_preprocessing.json"
        write_json(text_preprocessing_path, preprocess.stats)
        embedding_source_column = "text_cleaned"

    split.structured_df.to_parquet(structured_path, index=False)
    text_df.to_parquet(text_path, index=False)

    validation_report_path = config.paths.artifacts_reports_dir / "data_validation.json"
    column_types_path = config.paths.artifacts_reports_dir / "column_types.json"

    write_json(validation_report_path, validation_report)
    write_json(column_types_path, column_types.as_dict())

    eda_result = run_eda(
        structured_df=split.structured_df,
        column_types=column_types,
        plots_dir=config.paths.artifacts_plots_dir,
        reports_dir=config.paths.artifacts_reports_dir,
    )

    embeddings_path: Path | None = None
    embedding_meta_path: Path | None = None
    text_clusters_path: Path | None = None
    cluster_interpretations_path: Path | None = None
    final_dataset_path: Path | None = None
    cluster_correlations_path: Path | None = None
    final_report_json_path: Path | None = None
    final_report_markdown_path: Path | None = None
    final_report_context_path: Path | None = None
    if config.embeddings.enabled:
        embedding_result = build_embeddings(
            text_df=text_df,
            text_column=embedding_source_column,
            config=config.embeddings,
            models_dir=config.paths.artifacts_models_dir,
            reports_dir=config.paths.artifacts_reports_dir,
        )
        embeddings_path = embedding_result.embeddings_path
        embedding_meta_path = embedding_result.metadata_path

    if config.clustering.enabled:
        if embeddings_path is None:
            raise ValueError("Clustering requires embeddings. Set embeddings.enabled=true.")
        embeddings_array = np.load(embeddings_path)
        clustering = run_clustering(
            text_df=text_df,
            embeddings=embeddings_array,
            config=config.clustering,
            sentiment_cfg=config.sentiment,
            reports_dir=config.paths.artifacts_reports_dir,
        )
        text_df = clustering.text_df
        text_df.to_parquet(text_path, index=False)
        text_clusters_path = clustering.report_path
        if clustering.quality_status == "low":
            raise ValueError(
                f"Cluster quality gate failed: silhouette={clustering.silhouette_score}, "
                f"threshold={config.clustering.min_silhouette_score}"
            )

    if config.cluster_interpretation.enabled and "cluster_id" in text_df.columns:
        interpretation = run_cluster_interpretation(
            text_df=text_df,
            gigachat_cfg=config.gigachat,
            interpretation_cfg=config.cluster_interpretation,
            reports_dir=config.paths.artifacts_reports_dir,
            logs_dir=config.paths.artifacts_logs_dir,
        )
        cluster_interpretations_path = interpretation.report_path

    if config.merge_analysis.enabled and "cluster_id" in text_df.columns:
        merge_result = merge_and_analyze(
            structured_df=split.structured_df,
            text_df=text_df,
            column_types=column_types,
            cfg=config.merge_analysis,
            processed_dir=config.paths.data_processed_dir,
            reports_dir=config.paths.artifacts_reports_dir,
        )
        final_dataset_path = merge_result.final_dataset_path
        cluster_correlations_path = merge_result.report_path

    if config.final_report.enabled:
        report_result = generate_final_report(
            reports_dir=config.paths.artifacts_reports_dir,
            gigachat_cfg=config.gigachat,
            report_cfg=config.final_report,
            logs_dir=config.paths.artifacts_logs_dir,
        )
        final_report_json_path = report_result.json_path
        final_report_markdown_path = report_result.markdown_path
        final_report_context_path = report_result.context_path

    return RunSummary(
        input_rows=int(df.shape[0]),
        input_columns=int(df.shape[1]),
        id_column=id_column,
        structured_path=structured_path,
        text_path=text_path,
        validation_report_path=validation_report_path,
        column_types_path=column_types_path,
        eda_summary_path=eda_result.eda_summary_path,
        outlier_flags_path=eda_result.outlier_flags_path,
        text_preprocessing_path=text_preprocessing_path,
        embeddings_path=embeddings_path,
        embedding_meta_path=embedding_meta_path,
        text_clusters_path=text_clusters_path,
        cluster_interpretations_path=cluster_interpretations_path,
        final_dataset_path=final_dataset_path,
        cluster_correlations_path=cluster_correlations_path,
        final_report_json_path=final_report_json_path,
        final_report_markdown_path=final_report_markdown_path,
        final_report_context_path=final_report_context_path,
    )
