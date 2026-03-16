from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    input_dataset: Path
    data_raw_dir: Path
    data_interim_dir: Path
    data_processed_dir: Path
    artifacts_reports_dir: Path
    artifacts_plots_dir: Path
    artifacts_models_dir: Path
    artifacts_logs_dir: Path


@dataclass(frozen=True)
class ColumnsConfig:
    id_column: str | None
    manual_numeric: list[str]
    manual_categorical: list[str]
    manual_text: list[str]
    text_min_avg_length: int
    text_min_max_length: int


@dataclass(frozen=True)
class SplitConfig:
    merge_text_columns: bool
    merged_text_column_name: str
    merge_separator: str


@dataclass(frozen=True)
class TextProcessingConfig:
    enabled: bool
    source_column: str
    lowercase: bool
    remove_punctuation: bool
    normalize_whitespace: bool
    stemming_mode: str
    min_russian_char_ratio: float


@dataclass(frozen=True)
class EmbeddingsConfig:
    enabled: bool
    model_name: str
    batch_size: int
    normalize_embeddings: bool
    show_progress_bar: bool
    checkpoint_every: int


@dataclass(frozen=True)
class KMeansConfig:
    min_k: int
    max_k: int
    n_init: int


@dataclass(frozen=True)
class HdbscanConfig:
    min_cluster_size: int
    min_samples: int | None


@dataclass(frozen=True)
class ClusteringConfig:
    enabled: bool
    algorithm: str
    random_state: int
    kmeans: KMeansConfig
    hdbscan: HdbscanConfig
    tiny_cluster_threshold: int
    min_silhouette_score: float


@dataclass(frozen=True)
class SentimentConfig:
    enabled: bool


@dataclass(frozen=True)
class GigaChatConfig:
    enabled: bool
    base_url: str
    oauth_base_url: str
    oauth_endpoint: str
    chat_endpoint: str
    scope: str
    use_oauth: bool
    verify_ssl: bool
    model: str
    model_fallbacks: list[str]
    api_key: str
    api_key_env_var: str
    timeout_seconds: int
    max_retries: int
    backoff_seconds: float
    temperature: float
    safe_logging: bool
    log_raw_text: bool

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        return os.getenv(self.api_key_env_var, "")


@dataclass(frozen=True)
class ClusterInterpretationConfig:
    enabled: bool
    min_texts_per_cluster: int
    sample_texts_per_cluster: int
    max_text_length: int
    prompt_language: str


@dataclass(frozen=True)
class MergeAnalysisConfig:
    enabled: bool
    alpha: float
    min_group_size: int


@dataclass(frozen=True)
class FinalReportConfig:
    enabled: bool
    language: str
    max_key_findings: int
    include_methodology: bool


@dataclass(frozen=True)
class AppConfig:
    seed: int
    paths: PathsConfig
    columns: ColumnsConfig
    split: SplitConfig
    text_processing: TextProcessingConfig
    embeddings: EmbeddingsConfig
    clustering: ClusteringConfig
    sentiment: SentimentConfig
    gigachat: GigaChatConfig
    cluster_interpretation: ClusterInterpretationConfig
    merge_analysis: MergeAnalysisConfig
    final_report: FinalReportConfig
    project_root: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return data


def _to_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: str | Path) -> AppConfig:
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    raw = _read_yaml(config_file)
    root = config_file.parent.parent

    paths_raw = raw.get("paths", {})
    columns_raw = raw.get("columns", {})
    split_raw = raw.get("split", {})
    text_raw = raw.get("text_processing", {})
    emb_raw = raw.get("embeddings", {})
    clustering_raw = raw.get("clustering", {})
    kmeans_raw = clustering_raw.get("kmeans", {})
    hdbscan_raw = clustering_raw.get("hdbscan", {})
    sentiment_raw = raw.get("sentiment", {})
    gigachat_raw = raw.get("gigachat", {})
    interp_raw = raw.get("cluster_interpretation", {})
    merge_raw = raw.get("merge_analysis", {})
    final_report_raw = raw.get("final_report", {})

    paths = PathsConfig(
        input_dataset=_to_path(paths_raw["input_dataset"], root),
        data_raw_dir=_to_path(paths_raw["data_raw_dir"], root),
        data_interim_dir=_to_path(paths_raw["data_interim_dir"], root),
        data_processed_dir=_to_path(paths_raw["data_processed_dir"], root),
        artifacts_reports_dir=_to_path(paths_raw["artifacts_reports_dir"], root),
        artifacts_plots_dir=_to_path(paths_raw["artifacts_plots_dir"], root),
        artifacts_models_dir=_to_path(paths_raw["artifacts_models_dir"], root),
        artifacts_logs_dir=_to_path(paths_raw["artifacts_logs_dir"], root),
    )

    columns = ColumnsConfig(
        id_column=columns_raw.get("id_column"),
        manual_numeric=columns_raw.get("manual_numeric", []),
        manual_categorical=columns_raw.get("manual_categorical", []),
        manual_text=columns_raw.get("manual_text", []),
        text_min_avg_length=int(columns_raw.get("text_min_avg_length", 25)),
        text_min_max_length=int(columns_raw.get("text_min_max_length", 60)),
    )

    split = SplitConfig(
        merge_text_columns=bool(split_raw.get("merge_text_columns", True)),
        merged_text_column_name=str(split_raw.get("merged_text_column_name", "combined_text")),
        merge_separator=str(split_raw.get("merge_separator", " || ")),
    )

    text_processing = TextProcessingConfig(
        enabled=bool(text_raw.get("enabled", True)),
        source_column=str(text_raw.get("source_column", "combined_text")),
        lowercase=bool(text_raw.get("lowercase", True)),
        remove_punctuation=bool(text_raw.get("remove_punctuation", True)),
        normalize_whitespace=bool(text_raw.get("normalize_whitespace", True)),
        stemming_mode=str(text_raw.get("stemming_mode", "none")),
        min_russian_char_ratio=float(text_raw.get("min_russian_char_ratio", 0.35)),
    )

    embeddings = EmbeddingsConfig(
        enabled=bool(emb_raw.get("enabled", True)),
        model_name=str(emb_raw.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")),
        batch_size=int(emb_raw.get("batch_size", 32)),
        normalize_embeddings=bool(emb_raw.get("normalize_embeddings", True)),
        show_progress_bar=bool(emb_raw.get("show_progress_bar", True)),
        checkpoint_every=int(emb_raw.get("checkpoint_every", 50)),
    )

    clustering = ClusteringConfig(
        enabled=bool(clustering_raw.get("enabled", True)),
        algorithm=str(clustering_raw.get("algorithm", "kmeans")),
        random_state=int(clustering_raw.get("random_state", 42)),
        kmeans=KMeansConfig(
            min_k=int(kmeans_raw.get("min_k", 2)),
            max_k=int(kmeans_raw.get("max_k", 12)),
            n_init=int(kmeans_raw.get("n_init", 10)),
        ),
        hdbscan=HdbscanConfig(
            min_cluster_size=int(hdbscan_raw.get("min_cluster_size", 10)),
            min_samples=None if hdbscan_raw.get("min_samples", None) is None else int(hdbscan_raw.get("min_samples")),
        ),
        tiny_cluster_threshold=int(clustering_raw.get("tiny_cluster_threshold", 5)),
        min_silhouette_score=float(clustering_raw.get("min_silhouette_score", 0.05)),
    )

    sentiment = SentimentConfig(enabled=bool(sentiment_raw.get("enabled", False)))

    gigachat = GigaChatConfig(
        enabled=bool(gigachat_raw.get("enabled", False)),
        base_url=str(gigachat_raw.get("base_url", "https://gigachat.devices.sberbank.ru")),
        oauth_base_url=str(gigachat_raw.get("oauth_base_url", "https://ngw.devices.sberbank.ru:9443")),
        oauth_endpoint=str(gigachat_raw.get("oauth_endpoint", "/api/v2/oauth")),
        chat_endpoint=str(gigachat_raw.get("chat_endpoint", "/api/v1/chat/completions")),
        scope=str(gigachat_raw.get("scope", "GIGACHAT_API_PERS")),
        use_oauth=bool(gigachat_raw.get("use_oauth", True)),
        verify_ssl=bool(gigachat_raw.get("verify_ssl", True)),
        model=str(gigachat_raw.get("model", "GigaChat")),
        model_fallbacks=[str(m) for m in gigachat_raw.get("model_fallbacks", [])],
        api_key=str(gigachat_raw.get("api_key", "")),
        api_key_env_var=str(gigachat_raw.get("api_key_env_var", "GIGACHAT_API_KEY")),
        timeout_seconds=int(gigachat_raw.get("timeout_seconds", 60)),
        max_retries=int(gigachat_raw.get("max_retries", 3)),
        backoff_seconds=float(gigachat_raw.get("backoff_seconds", 1.5)),
        temperature=float(gigachat_raw.get("temperature", 0.2)),
        safe_logging=bool(gigachat_raw.get("safe_logging", True)),
        log_raw_text=bool(gigachat_raw.get("log_raw_text", False)),
    )

    cluster_interpretation = ClusterInterpretationConfig(
        enabled=bool(interp_raw.get("enabled", True)),
        min_texts_per_cluster=int(interp_raw.get("min_texts_per_cluster", 3)),
        sample_texts_per_cluster=int(interp_raw.get("sample_texts_per_cluster", 5)),
        max_text_length=int(interp_raw.get("max_text_length", 220)),
        prompt_language=str(interp_raw.get("prompt_language", "ru")),
    )

    merge_analysis = MergeAnalysisConfig(
        enabled=bool(merge_raw.get("enabled", True)),
        alpha=float(merge_raw.get("alpha", 0.05)),
        min_group_size=int(merge_raw.get("min_group_size", 3)),
    )

    final_report = FinalReportConfig(
        enabled=bool(final_report_raw.get("enabled", True)),
        language=str(final_report_raw.get("language", "ru")),
        max_key_findings=int(final_report_raw.get("max_key_findings", 10)),
        include_methodology=bool(final_report_raw.get("include_methodology", True)),
    )

    return AppConfig(
        seed=int(raw.get("seed", 42)),
        paths=paths,
        columns=columns,
        split=split,
        text_processing=text_processing,
        embeddings=embeddings,
        clustering=clustering,
        sentiment=sentiment,
        gigachat=gigachat,
        cluster_interpretation=cluster_interpretation,
        merge_analysis=merge_analysis,
        final_report=final_report,
        project_root=root,
    )
