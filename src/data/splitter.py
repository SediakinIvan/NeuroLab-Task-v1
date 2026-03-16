from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import SplitConfig
from src.data.validation import ColumnTypes


@dataclass(frozen=True)
class SplitResult:
    structured_df: pd.DataFrame
    text_df: pd.DataFrame


def split_dataset(df: pd.DataFrame, column_types: ColumnTypes, split_cfg: SplitConfig) -> SplitResult:
    id_col = column_types.id_column

    structured_columns = [id_col, *column_types.numeric, *column_types.categorical]
    text_columns = [id_col, *column_types.text]

    structured_df = df[structured_columns].copy()
    text_df = df[text_columns].copy()

    if split_cfg.merge_text_columns and column_types.text:
        merged_series = (
            text_df[column_types.text]
            .fillna("")
            .astype(str)
            .apply(lambda row: split_cfg.merge_separator.join([v for v in row if v.strip()]), axis=1)
        )
        text_df[split_cfg.merged_text_column_name] = merged_series

    return SplitResult(structured_df=structured_df, text_df=text_df)
