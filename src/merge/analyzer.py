from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal

from src.config import MergeAnalysisConfig
from src.data.validation import ColumnTypes
from src.io_utils import write_json


@dataclass(frozen=True)
class MergeAnalysisResult:
    final_dataset_path: Path
    report_path: Path


def _cramers_v(contingency: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(contingency.to_numpy())
    n = contingency.to_numpy().sum()
    if n <= 0:
        return 0.0
    r, k = contingency.shape
    denom = min(r - 1, k - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


def _eta_squared(groups: list[np.ndarray]) -> float:
    all_vals = np.concatenate(groups)
    overall_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 for g in groups)
    ss_total = ((all_vals - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


def _epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    if n <= k:
        return 0.0
    return float((H - k + 1) / (n - k))


def merge_and_analyze(
    structured_df: pd.DataFrame,
    text_df: pd.DataFrame,
    column_types: ColumnTypes,
    cfg: MergeAnalysisConfig,
    processed_dir: Path,
    reports_dir: Path,
) -> MergeAnalysisResult:
    id_col = column_types.id_column
    if id_col not in structured_df.columns or id_col not in text_df.columns:
        raise ValueError(f"ID column '{id_col}' must exist in both datasets.")

    merge_integrity = {
        "structured_rows": int(len(structured_df)),
        "text_rows": int(len(text_df)),
        "structured_unique_ids": int(structured_df[id_col].nunique()),
        "text_unique_ids": int(text_df[id_col].nunique()),
        "structured_duplicate_ids": int(structured_df[id_col].duplicated().sum()),
        "text_duplicate_ids": int(text_df[id_col].duplicated().sum()),
    }

    if merge_integrity["structured_duplicate_ids"] > 0 or merge_integrity["text_duplicate_ids"] > 0:
        raise ValueError("Duplicate IDs detected before merge.")

    merged = structured_df.merge(text_df, on=id_col, how="inner", suffixes=("", "_text"))
    merge_integrity["merged_rows"] = int(len(merged))
    merge_integrity["merged_unique_ids"] = int(merged[id_col].nunique())
    merge_integrity["dropped_rows_from_structured"] = int(len(structured_df) - len(merged))
    merge_integrity["dropped_rows_from_text"] = int(len(text_df) - len(merged))

    if merge_integrity["dropped_rows_from_structured"] > 0 or merge_integrity["dropped_rows_from_text"] > 0:
        raise ValueError("Merge integrity check failed: row drops detected between structured and text datasets.")

    if "cluster_id" not in merged.columns:
        raise ValueError("cluster_id not found after merge. Run clustering stage first.")

    categorical_results: list[dict[str, Any]] = []
    for col in column_types.categorical:
        if col not in merged.columns:
            continue
        ct = pd.crosstab(merged["cluster_id"], merged[col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p_value, dof, expected = chi2_contingency(ct.to_numpy())
        categorical_results.append(
            {
                "feature": col,
                "test": "chi2_contingency",
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": _cramers_v(ct),
                "is_significant": bool(p_value < cfg.alpha),
                "cluster_levels": int(ct.shape[0]),
                "feature_levels": int(ct.shape[1]),
            }
        )

    numeric_results: list[dict[str, Any]] = []
    grouped = merged.groupby("cluster_id")
    for col in column_types.numeric:
        if col not in merged.columns:
            continue
        arrays = []
        for _, group in grouped:
            values = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy()
            if len(values) >= cfg.min_group_size:
                arrays.append(values)

        if len(arrays) < 2:
            continue

        anova_stat, anova_p = f_oneway(*arrays)
        kruskal_stat, kruskal_p = kruskal(*arrays)
        eta_sq = _eta_squared(arrays)
        eps_sq = _epsilon_squared_kruskal(float(kruskal_stat), int(sum(len(a) for a in arrays)), len(arrays))
        numeric_results.append(
            {
                "feature": col,
                "anova_f_statistic": float(anova_stat),
                "anova_p_value": float(anova_p),
                "kruskal_h_statistic": float(kruskal_stat),
                "kruskal_p_value": float(kruskal_p),
                "eta_squared": eta_sq,
                "epsilon_squared": eps_sq,
                "anova_significant": bool(anova_p < cfg.alpha),
                "kruskal_significant": bool(kruskal_p < cfg.alpha),
                "groups_used": len(arrays),
            }
        )

    final_dataset_path = processed_dir / "final_dataset.parquet"
    processed_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(final_dataset_path, index=False)

    report = {
        "alpha": cfg.alpha,
        "merge_integrity": merge_integrity,
        "categorical_associations": categorical_results,
        "numeric_cluster_comparisons": numeric_results,
    }
    report_path = reports_dir / "cluster_correlations.json"
    write_json(report_path, report)

    return MergeAnalysisResult(final_dataset_path=final_dataset_path, report_path=report_path)
