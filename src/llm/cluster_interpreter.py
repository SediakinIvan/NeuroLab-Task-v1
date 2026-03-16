from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import ValidationError, validate

from src.config import ClusterInterpretationConfig, GigaChatConfig
from src.io_utils import write_json
from src.llm.gigachat_client import GigaChatClient

INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cluster_id": {"type": "integer"},
        "cluster_name": {"type": "string"},
        "shared_patterns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "risks": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cluster_id", "cluster_name", "shared_patterns", "confidence", "risks", "evidence"],
}

RU_STOPWORDS = {
    "и",
    "в",
    "на",
    "с",
    "по",
    "но",
    "это",
    "для",
    "что",
    "как",
    "из",
    "к",
    "я",
    "мы",
    "он",
    "она",
    "они",
    "а",
    "у",
    "же",
    "не",
    "то",
    "от",
}


@dataclass(frozen=True)
class ClusterInterpretationResult:
    report_path: Path


def _top_terms(texts: list[str], n: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in text.lower().split():
            if len(token) < 3 or token in RU_STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]]


def _prompt_for_cluster(cluster_id: int, n_rows: int, terms: list[str], snippets: list[str], lang: str) -> str:
    return (
        "You must return strict JSON with keys: cluster_id, cluster_name, shared_patterns, "
        "confidence, risks, evidence.\n"
        "cluster_id must be integer. confidence must be one of: low, medium, high.\n"
        f"Preferred language for textual values: {lang}.\n\n"
        f"Cluster ID: {cluster_id}\n"
        f"Respondent count: {n_rows}\n"
        f"Top terms: {terms}\n"
        f"Sample respondent texts:\n- " + "\n- ".join(snippets)
    )


def _fallback_interpretation(cluster_id: int, n_rows: int, terms: list[str], snippets: list[str]) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "cluster_name": f"Segment {cluster_id}",
        "shared_patterns": [
            f"Participants share recurring themes around: {', '.join(terms[:4])}" if terms else "Shared text patterns detected",
            f"Group size is {n_rows} respondents",
        ],
        "confidence": "low",
        "risks": ["Automatic fallback used because LLM output was unavailable or invalid"],
        "evidence": snippets[:3],
    }


def _cache_key(payload: dict[str, Any]) -> str:
    data = str(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize_interpretation(candidate: dict[str, Any], cluster_id: int) -> dict[str, Any]:
    normalized = dict(candidate)
    normalized["cluster_id"] = int(normalized.get("cluster_id", cluster_id))

    for key in ("cluster_name",):
        if key in normalized and not isinstance(normalized[key], str):
            normalized[key] = str(normalized[key])

    for key in ("shared_patterns", "risks"):
        value = normalized.get(key, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = [str(value)]
        normalized[key] = [str(v) for v in value]

    conf = str(normalized.get("confidence", "low")).lower().strip()
    if conf not in {"low", "medium", "high"}:
        conf = "low"
    normalized["confidence"] = conf

    evidence = normalized.get("evidence", [])
    if isinstance(evidence, dict):
        flattened = []
        for k, v in evidence.items():
            flattened.append(f"{k}: {v}")
        evidence = flattened
    elif isinstance(evidence, str):
        evidence = [evidence]
    elif not isinstance(evidence, list):
        evidence = [str(evidence)]
    normalized["evidence"] = [str(v) for v in evidence]
    return normalized


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        return {}
    except Exception:
        return {}


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_cluster_interpretation(
    text_df: pd.DataFrame,
    gigachat_cfg: GigaChatConfig,
    interpretation_cfg: ClusterInterpretationConfig,
    reports_dir: Path,
    logs_dir: Path,
) -> ClusterInterpretationResult:
    if "cluster_id" not in text_df.columns:
        raise ValueError("cluster_id column is required for cluster interpretation.")

    text_col = "text_cleaned" if "text_cleaned" in text_df.columns else "combined_text"
    client = GigaChatClient(gigachat_cfg, logs_dir=logs_dir)

    cache_file = logs_dir / "gigachat_cache.json"
    cache = _load_cache(cache_file)

    clusters_output: list[dict[str, Any]] = []
    for cluster_id, group in text_df.groupby("cluster_id"):
        cluster_id = int(cluster_id)
        texts = group[text_col].fillna("").astype(str).tolist()
        if len(texts) < interpretation_cfg.min_texts_per_cluster:
            continue

        snippets = [t[: interpretation_cfg.max_text_length] for t in texts[: interpretation_cfg.sample_texts_per_cluster]]
        terms = _top_terms(texts)
        prompt = _prompt_for_cluster(
            cluster_id=cluster_id,
            n_rows=len(group),
            terms=terms,
            snippets=snippets,
            lang=interpretation_cfg.prompt_language,
        )
        fingerprint = _cache_key(
            {
                "cluster_id": cluster_id,
                "n_rows": len(group),
                "terms": tuple(terms),
                "snippets": tuple(snippets),
                "model": gigachat_cfg.model,
            }
        )

        if fingerprint in cache:
            interpretation = cache[fingerprint]
            source = "cache"
        else:
            source = "fallback"
            interpretation = _fallback_interpretation(cluster_id, len(group), terms, snippets)
            if client.is_ready:
                try:
                    candidate = client.chat_json(prompt=prompt, request_id=f"cluster-{cluster_id}-{fingerprint[:12]}")
                    candidate = _normalize_interpretation(candidate, cluster_id=cluster_id)
                    validate(instance=candidate, schema=INTERPRETATION_SCHEMA)
                    interpretation = candidate
                    source = "gigachat"
                except (RuntimeError, ValidationError):
                    interpretation = _fallback_interpretation(cluster_id, len(group), terms, snippets)
                    source = "fallback"
            cache[fingerprint] = interpretation

        interpretation["cluster_id"] = cluster_id
        interpretation["_source"] = source
        clusters_output.append(interpretation)

    _save_cache(cache_file, cache)
    report_payload = {
        "enabled": interpretation_cfg.enabled,
        "gigachat_enabled": gigachat_cfg.enabled,
        "api_ready": client.is_ready,
        "clusters_interpreted": len(clusters_output),
        "items": clusters_output,
    }
    report_path = reports_dir / "cluster_interpretations.json"
    write_json(report_path, report_payload)
    return ClusterInterpretationResult(report_path=report_path)
