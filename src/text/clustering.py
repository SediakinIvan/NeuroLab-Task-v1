from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import ClusteringConfig, SentimentConfig
from src.io_utils import write_json

try:
    import hdbscan
except ImportError:  # pragma: no cover
    hdbscan = None


POSITIVE_RU = {
    "помогает",
    "польза",
    "вдохновляет",
    "развиваться",
    "интересно",
    "нравится",
    "хорошо",
    "поддержка",
    "полезно",
}

NEGATIVE_RU = {
    "мешает",
    "плохо",
    "стресс",
    "тревога",
    "зависаю",
    "зависимость",
    "конфликт",
    "ссор",
    "негатив",
    "отвлекает",
    "страдает",
    "невысыпаюсь",
    "хейт",
}


@dataclass(frozen=True)
class ClusterResult:
    text_df: pd.DataFrame
    report_path: Path
    quality_status: str
    silhouette_score: float | None


def _compute_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    unique = sorted(set(labels.tolist()))
    if -1 in unique:
        # Exclude noise for quality metrics.
        mask = labels != -1
        X_eval = X[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X
        labels_eval = labels

    if len(X_eval) < 3:
        return None, None
    n_clusters = len(set(labels_eval.tolist()))
    if n_clusters < 2:
        return None, None

    sil = float(silhouette_score(X_eval, labels_eval))
    dbi = float(davies_bouldin_score(X_eval, labels_eval))
    return sil, dbi


def _run_kmeans(X: np.ndarray, cfg: ClusteringConfig) -> tuple[np.ndarray, dict[str, Any]]:
    min_k = max(2, cfg.kmeans.min_k)
    max_k = min(cfg.kmeans.max_k, max(2, len(X) - 1))
    if min_k > max_k:
        min_k = max_k

    candidates: list[dict[str, Any]] = []
    best_labels: np.ndarray | None = None
    best_score = float("-inf")
    best_k = min_k

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=cfg.random_state,
            n_init=cfg.kmeans.n_init,
        )
        labels = model.fit_predict(X)
        sil, dbi = _compute_metrics(X, labels)
        composite = (sil if sil is not None else -1.0) - (dbi if dbi is not None else 10.0) * 0.05
        candidates.append(
            {
                "k": k,
                "silhouette_score": sil,
                "davies_bouldin_score": dbi,
                "composite_score": float(composite),
                "inertia": float(model.inertia_),
            }
        )
        if composite > best_score:
            best_score = composite
            best_labels = labels
            best_k = k

    assert best_labels is not None
    return best_labels, {
        "algorithm": "kmeans",
        "selected_k": best_k,
        "candidate_scores": candidates,
    }


def _run_hdbscan(X: np.ndarray, cfg: ClusteringConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if hdbscan is None:
        raise ImportError("hdbscan is not installed.")

    model = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
    )
    labels = model.fit_predict(X)
    return labels, {
        "algorithm": "hdbscan",
        "min_cluster_size": cfg.hdbscan.min_cluster_size,
        "min_samples": cfg.hdbscan.min_samples,
        "noise_points": int((labels == -1).sum()),
    }


def _cluster_confidence_by_size(labels: np.ndarray, tiny_cluster_threshold: int) -> dict[int, str]:
    counts = pd.Series(labels).value_counts().to_dict()
    confidence: dict[int, str] = {}
    for cluster_id, count in counts.items():
        if cluster_id == -1:
            confidence[int(cluster_id)] = "low"
        elif count < tiny_cluster_threshold:
            confidence[int(cluster_id)] = "low"
        else:
            confidence[int(cluster_id)] = "normal"
    return confidence


def _sentiment_label(text: str) -> tuple[str, float]:
    tokens = set(text.lower().split())
    pos = len(tokens & POSITIVE_RU)
    neg = len(tokens & NEGATIVE_RU)
    score = float(pos - neg)
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", score


def run_clustering(
    text_df: pd.DataFrame,
    embeddings: np.ndarray,
    config: ClusteringConfig,
    sentiment_cfg: SentimentConfig,
    reports_dir: Path,
) -> ClusterResult:
    if len(text_df) != len(embeddings):
        raise ValueError("Text rows and embedding rows must match.")

    if config.algorithm == "hdbscan":
        labels, algo_meta = _run_hdbscan(embeddings, config)
    else:
        labels, algo_meta = _run_kmeans(embeddings, config)

    sil, dbi = _compute_metrics(embeddings, labels)
    cluster_counts = pd.Series(labels).value_counts().sort_index().to_dict()
    low_confidence_by_cluster = _cluster_confidence_by_size(labels, config.tiny_cluster_threshold)

    result_df = text_df.copy()
    result_df["cluster_id"] = labels.astype(int)
    result_df["cluster_confidence"] = result_df["cluster_id"].map(low_confidence_by_cluster).fillna("normal")
    result_df["is_low_confidence_cluster"] = result_df["cluster_confidence"] == "low"

    if sentiment_cfg.enabled:
        if "text_cleaned" in result_df.columns:
            source_col = "text_cleaned"
        elif "combined_text" in result_df.columns:
            source_col = "combined_text"
        else:
            raise ValueError("No suitable text column found for sentiment scoring.")
        sentiments = result_df[source_col].fillna("").astype(str).apply(_sentiment_label)
        result_df["sentiment_label"] = sentiments.apply(lambda x: x[0])
        result_df["sentiment_score"] = sentiments.apply(lambda x: x[1])

    non_noise_counts = {k: v for k, v in cluster_counts.items() if int(k) != -1}
    size_balance = None
    if non_noise_counts:
        min_size = min(non_noise_counts.values())
        max_size = max(non_noise_counts.values())
        if min_size > 0:
            size_balance = float(max_size / min_size)

    quality_status = "ok"
    if sil is not None and sil < config.min_silhouette_score:
        quality_status = "low"

    report = {
        "algorithm_meta": algo_meta,
        "n_rows": int(len(result_df)),
        "n_clusters_including_noise": int(len(set(labels.tolist()))),
        "silhouette_score": sil,
        "davies_bouldin_score": dbi,
        "cluster_size_balance_max_to_min": size_balance,
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "low_confidence_clusters": [int(k) for k, v in low_confidence_by_cluster.items() if v == "low"],
        "quality_status": quality_status,
        "sentiment_enabled": sentiment_cfg.enabled,
    }

    report_path = reports_dir / "text_clusters.json"
    write_json(report_path, report)
    return ClusterResult(
        text_df=result_df,
        report_path=report_path,
        quality_status=quality_status,
        silhouette_score=sil,
    )
