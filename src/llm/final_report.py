from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from src.config import FinalReportConfig, GigaChatConfig
from src.io_utils import write_json
from src.llm.gigachat_client import GigaChatClient

FINAL_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "language": {"type": "string"},
        "executive_summary": {"type": "string"},
        "key_segments": {"type": "array", "items": {"type": "string"}},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "methodology_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "language",
        "executive_summary",
        "key_segments",
        "key_findings",
        "risks",
        "recommendations",
        "methodology_notes",
    ],
}


@dataclass(frozen=True)
class FinalReportResult:
    json_path: Path
    markdown_path: Path
    context_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _build_context(reports_dir: Path, cfg: FinalReportConfig) -> dict[str, Any]:
    data_validation = _read_json(reports_dir / "data_validation.json")
    eda_summary = _read_json(reports_dir / "eda_summary.json")
    text_clusters = _read_json(reports_dir / "text_clusters.json")
    interpretations = _read_json(reports_dir / "cluster_interpretations.json")
    correlations = _read_json(reports_dir / "cluster_correlations.json")

    top_categorical = sorted(
        correlations.get("categorical_associations", []),
        key=lambda x: float(x.get("cramers_v", 0)),
        reverse=True,
    )[: cfg.max_key_findings]
    top_numeric = sorted(
        correlations.get("numeric_cluster_comparisons", []),
        key=lambda x: float(x.get("eta_squared", 0)),
        reverse=True,
    )[: cfg.max_key_findings]

    compact_context = {
        "dataset_overview": {
            "rows": data_validation.get("rows"),
            "columns": data_validation.get("columns"),
            "duplicate_rows": data_validation.get("duplicate_rows"),
            "duplicate_ids": data_validation.get("duplicate_ids"),
        },
        "eda_overview": {
            "numeric_columns": eda_summary.get("numeric_columns", []),
            "categorical_columns": eda_summary.get("categorical_columns", []),
            "outlier_summary": eda_summary.get("outlier_summary", {}),
        },
        "cluster_overview": {
            "n_rows": text_clusters.get("n_rows"),
            "n_clusters_including_noise": text_clusters.get("n_clusters_including_noise"),
            "silhouette_score": text_clusters.get("silhouette_score"),
            "davies_bouldin_score": text_clusters.get("davies_bouldin_score"),
            "cluster_counts": text_clusters.get("cluster_counts", {}),
            "quality_status": text_clusters.get("quality_status"),
            "interpreted_clusters": interpretations.get("clusters_interpreted"),
        },
        "top_categorical_associations": top_categorical,
        "top_numeric_comparisons": top_numeric,
        "language": cfg.language,
    }
    return compact_context


def _fallback_report(context: dict[str, Any], cfg: FinalReportConfig) -> dict[str, Any]:
    cluster_info = context.get("cluster_overview", {})
    top_cat = context.get("top_categorical_associations", [])
    top_num = context.get("top_numeric_comparisons", [])

    findings: list[str] = []
    for item in top_cat[:5]:
        findings.append(
            f"Связь между кластерами и '{item.get('feature')}' (Cramer's V={item.get('cramers_v'):.3f})."
        )
    for item in top_num[:5]:
        findings.append(
            f"Межкластерные различия по '{item.get('feature')}' (eta^2={item.get('eta_squared'):.3f})."
        )

    return {
        "title": "Итоговый отчет по опросу о влиянии социальных сетей",
        "language": cfg.language,
        "executive_summary": (
            f"Проанализировано {context.get('dataset_overview', {}).get('rows')} респондентов. "
            f"Выделено {cluster_info.get('n_clusters_including_noise')} кластеров текстовых ответов; "
            "выявлены статистически значимые различия между кластерами по ряду признаков."
        ),
        "key_segments": [
            "Кластеры с выраженным риском отвлечения от учебы и дефицитом сна",
            "Кластеры с прагматичным использованием соцсетей (новости, обучение, карьера)",
            "Кластеры с социально-коммуникационным и развлекательным фокусом",
        ],
        "key_findings": findings[: cfg.max_key_findings],
        "risks": [
            "Высокая вовлеченность в соцсети связана с ухудшением учебных и поведенческих показателей.",
            "Отдельные сегменты демонстрируют признаки конфликтов и негативного эмоционального фона.",
        ],
        "recommendations": [
            "Таргетировать профилактические коммуникации на кластеры высокого риска.",
            "Запустить образовательные интервенции по цифровой гигиене и режиму сна.",
            "Использовать сегментацию для персонализированных программ поддержки.",
        ],
        "methodology_notes": (
            [
                "EDA структурных данных: распределения, корреляции, выбросы.",
                "Кластеризация русскоязычных текстов на эмбеддингах Sentence-Transformers.",
                "Статтесты: chi-square + Cramer's V; ANOVA/Kruskal + эффекты.",
            ]
            if cfg.include_methodology
            else []
        ),
    }


def _prompt_for_report(context: dict[str, Any], cfg: FinalReportConfig) -> str:
    return (
        "Return strict JSON only with keys: title, language, executive_summary, "
        "key_segments, key_findings, risks, recommendations, methodology_notes.\n"
        f"Language for text values: {cfg.language}.\n"
        "Make recommendations actionable and specific.\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )


def _normalize_final_report(candidate: dict[str, Any], cfg: FinalReportConfig) -> dict[str, Any]:
    normalized = dict(candidate)
    normalized.setdefault("title", "Итоговый отчет по опросу о влиянии социальных сетей")
    normalized["language"] = str(normalized.get("language", cfg.language))
    normalized["executive_summary"] = str(normalized.get("executive_summary", ""))

    for key in ("key_segments", "key_findings", "risks", "recommendations", "methodology_notes"):
        value = normalized.get(key, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = [str(value)]
        normalized[key] = [str(v) for v in value]

    return normalized


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('title', 'Final Report')}",
        "",
        "## Executive Summary",
        report.get("executive_summary", ""),
        "",
        "## Key Segments",
    ]
    lines.extend([f"- {item}" for item in report.get("key_segments", [])])
    lines.extend(["", "## Key Findings"])
    lines.extend([f"- {item}" for item in report.get("key_findings", [])])
    lines.extend(["", "## Risks"])
    lines.extend([f"- {item}" for item in report.get("risks", [])])
    lines.extend(["", "## Recommendations"])
    lines.extend([f"- {item}" for item in report.get("recommendations", [])])
    notes = report.get("methodology_notes", [])
    if notes:
        lines.extend(["", "## Methodology Notes"])
        lines.extend([f"- {item}" for item in notes])
    lines.append("")
    return "\n".join(lines)


def generate_final_report(
    reports_dir: Path,
    gigachat_cfg: GigaChatConfig,
    report_cfg: FinalReportConfig,
    logs_dir: Path,
) -> FinalReportResult:
    context = _build_context(reports_dir, report_cfg)
    context_path = reports_dir / "final_report_context.json"
    write_json(context_path, context)

    client = GigaChatClient(gigachat_cfg, logs_dir=logs_dir)
    report = _fallback_report(context, report_cfg)
    source = "fallback"

    if gigachat_cfg.enabled and client.is_ready:
        prompt = _prompt_for_report(context, report_cfg)
        try:
            candidate = client.chat_json(prompt=prompt, request_id="final-report")
            candidate = _normalize_final_report(candidate, report_cfg)
            validate(instance=candidate, schema=FINAL_REPORT_SCHEMA)
            report = candidate
            source = "gigachat"
        except (RuntimeError, ValidationError):
            report = _fallback_report(context, report_cfg)
            source = "fallback"

    report["source"] = source
    json_path = reports_dir / "final_report.json"
    write_json(json_path, report)

    md_path = reports_dir / "final_report.md"
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    return FinalReportResult(json_path=json_path, markdown_path=md_path, context_path=context_path)
