from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import ColumnsConfig


@dataclass(frozen=True)
class ColumnTypes:
    id_column: str
    numeric: list[str]
    categorical: list[str]
    text: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id_column": self.id_column,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "text": self.text,
        }


def ensure_stable_id(df: pd.DataFrame, id_column: str | None) -> tuple[pd.DataFrame, str]:
    work_df = df.copy()
    if id_column and id_column in work_df.columns:
        return work_df, id_column

    generated_id = "__row_id"
    if generated_id in work_df.columns:
        generated_id = "__row_id_generated"
    work_df[generated_id] = range(1, len(work_df) + 1)
    return work_df, generated_id


def infer_column_types(df: pd.DataFrame, columns_cfg: ColumnsConfig, id_column: str) -> ColumnTypes:
    manual_numeric = set(columns_cfg.manual_numeric)
    manual_categorical = set(columns_cfg.manual_categorical)
    manual_text = set(columns_cfg.manual_text)

    numeric: list[str] = []
    categorical: list[str] = []
    text: list[str] = []

    for column in df.columns:
        if column == id_column:
            continue

        if column in manual_numeric:
            numeric.append(column)
            continue
        if column in manual_categorical:
            categorical.append(column)
            continue
        if column in manual_text:
            text.append(column)
            continue

        series = df[column]
        if is_numeric_dtype(series):
            numeric.append(column)
            continue

        non_null = series.dropna().astype(str).str.strip()
        if non_null.empty:
            categorical.append(column)
            continue

        avg_len = float(non_null.str.len().mean())
        max_len = int(non_null.str.len().max())
        is_text_like = (
            avg_len >= columns_cfg.text_min_avg_length or max_len >= columns_cfg.text_min_max_length
        )
        if is_text_like:
            text.append(column)
        else:
            categorical.append(column)

    return ColumnTypes(
        id_column=id_column,
        numeric=sorted(numeric),
        categorical=sorted(categorical),
        text=sorted(text),
    )


def build_validation_report(df: pd.DataFrame, id_column: str, column_types: ColumnTypes) -> dict[str, Any]:
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[id_column].duplicated().sum())
    null_ratio = {col: float(df[col].isna().mean()) for col in df.columns}

    impossible_values: dict[str, int] = {}
    for col in column_types.numeric:
        # Generic rule: survey metrics should not be negative.
        impossible_values[col] = int((df[col] < 0).sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "id_column": id_column,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "null_ratio": null_ratio,
        "impossible_negative_values": impossible_values,
    }
