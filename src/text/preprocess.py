from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import TextProcessingConfig

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


@dataclass(frozen=True)
class TextPreprocessResult:
    text_df: pd.DataFrame
    stats: dict[str, Any]


def _light_russian_stem(token: str) -> str:
    # Lightweight fallback stemmer for Russian-like suffix reduction.
    suffixes = (
        "иями",
        "ями",
        "ами",
        "ого",
        "ему",
        "ому",
        "ыми",
        "ими",
        "иях",
        "ах",
        "ях",
        "ия",
        "ья",
        "иям",
        "ий",
        "ый",
        "ой",
        "ая",
        "яя",
        "ое",
        "ее",
        "ам",
        "ям",
        "ов",
        "ев",
        "ом",
        "ем",
        "ам",
        "ям",
        "ах",
        "ях",
        "ы",
        "и",
        "а",
        "я",
        "о",
        "е",
        "у",
        "ю",
    )
    for suffix in suffixes:
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _russian_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters_only = [ch for ch in text if ch.isalpha()]
    if not letters_only:
        return 0.0
    ru_chars = len(_CYRILLIC_RE.findall("".join(letters_only)))
    return ru_chars / len(letters_only)


def _clean_text(value: str, cfg: TextProcessingConfig) -> str:
    text = value or ""
    if cfg.lowercase:
        text = text.lower()
    if cfg.remove_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    if cfg.normalize_whitespace:
        text = _SPACE_RE.sub(" ", text).strip()
    if cfg.stemming_mode == "light_ru":
        tokens = [_light_russian_stem(token) for token in text.split(" ") if token]
        text = " ".join(tokens)
    return text


def preprocess_text_dataframe(df: pd.DataFrame, cfg: TextProcessingConfig) -> TextPreprocessResult:
    if cfg.source_column not in df.columns:
        raise ValueError(
            f"Configured text source column '{cfg.source_column}' does not exist in text dataset."
        )

    work = df.copy()
    raw = work[cfg.source_column].fillna("").astype(str)
    cleaned = raw.apply(lambda value: _clean_text(value, cfg))
    ru_ratio = cleaned.apply(_russian_char_ratio)
    is_russian = ru_ratio >= cfg.min_russian_char_ratio

    work["text_cleaned"] = cleaned
    work["ru_char_ratio"] = ru_ratio
    work["is_russian_text"] = is_russian

    stats = {
        "rows_total": int(len(work)),
        "source_column": cfg.source_column,
        "non_empty_cleaned_rows": int((work["text_cleaned"].str.len() > 0).sum()),
        "russian_rows": int(work["is_russian_text"].sum()),
        "non_russian_rows": int((~work["is_russian_text"]).sum()),
        "mean_ru_char_ratio": float(work["ru_char_ratio"].mean()),
        "stemming_mode": cfg.stemming_mode,
    }
    return TextPreprocessResult(text_df=work, stats=stats)
