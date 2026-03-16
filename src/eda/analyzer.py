from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.validation import ColumnTypes
from src.io_utils import write_json


@dataclass(frozen=True)
class EdaResult:
    eda_summary_path: Path
    outlier_flags_path: Path
    generated_plots: list[str]


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


def _plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list[str], output_path: Path) -> None:
    if not numeric_cols:
        return
    cols_per_row = 3
    rows = math.ceil(len(numeric_cols) / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3.5 * rows))
    axes_array = np.array(axes).reshape(-1)

    for idx, col in enumerate(numeric_cols):
        ax = axes_array[idx]
        values = df[col].dropna()
        ax.hist(values, bins=20, color="#4C78A8", edgecolor="black", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    for idx in range(len(numeric_cols), len(axes_array)):
        axes_array[idx].axis("off")

    fig.suptitle("Numeric Feature Distributions", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_missingness_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    mask = df.isna().astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mask, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Missingness Heatmap (1 = missing)")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], output_path: Path) -> None:
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Numeric Correlation Heatmap")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_categorical_frequencies(df: pd.DataFrame, cat_cols: list[str], plots_dir: Path) -> list[str]:
    generated: list[str] = []
    for col in cat_cols:
        counts = df[col].astype(str).value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax, color="#72B7B2")
        ax.set_title(f"Categorical Frequency: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()

        filename = f"categorical_frequencies_{_safe_name(col)}.png"
        path = plots_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append(filename)
    return generated


def _compute_outlier_flags(df: pd.DataFrame, id_col: str, numeric_cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    outlier_df = pd.DataFrame({id_col: df[id_col]})
    summary: dict[str, Any] = {}

    any_outlier = np.zeros(len(df), dtype=bool)
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_flag = (series < lower) | (series > upper)

        median = series.median()
        mad = np.median(np.abs(series.dropna() - median))
        if mad == 0:
            robust_z = pd.Series(np.zeros(len(series)), index=series.index)
        else:
            robust_z = 0.6745 * (series - median) / mad
        rz_flag = robust_z.abs() > 3.5

        combined_flag = (iqr_flag | rz_flag).fillna(False)
        any_outlier = any_outlier | combined_flag.to_numpy()

        outlier_df[f"{col}__iqr_outlier"] = iqr_flag.fillna(False)
        outlier_df[f"{col}__robust_z_outlier"] = rz_flag.fillna(False)
        outlier_df[f"{col}__outlier"] = combined_flag

        summary[col] = {
            "iqr_outlier_count": int(iqr_flag.fillna(False).sum()),
            "robust_z_outlier_count": int(rz_flag.fillna(False).sum()),
            "combined_outlier_count": int(combined_flag.sum()),
            "iqr_bounds": {"lower": float(lower), "upper": float(upper)},
            "median": float(median),
            "mad": float(mad),
        }

    outlier_df["any_outlier"] = any_outlier
    return outlier_df, summary


def run_eda(
    structured_df: pd.DataFrame,
    column_types: ColumnTypes,
    plots_dir: Path,
    reports_dir: Path,
) -> EdaResult:
    numeric_cols = [c for c in column_types.numeric if c in structured_df.columns]
    cat_cols = [c for c in column_types.categorical if c in structured_df.columns]
    id_col = column_types.id_column

    generated_plots: list[str] = []

    numeric_plot = "numeric_distributions.png"
    _plot_numeric_distributions(structured_df, numeric_cols, plots_dir / numeric_plot)
    if numeric_cols:
        generated_plots.append(numeric_plot)

    missing_plot = "missingness_heatmap.png"
    _plot_missingness_heatmap(structured_df, plots_dir / missing_plot)
    generated_plots.append(missing_plot)

    corr_plot = "correlation_heatmap.png"
    _plot_correlation_heatmap(structured_df, numeric_cols, plots_dir / corr_plot)
    if len(numeric_cols) >= 2:
        generated_plots.append(corr_plot)

    generated_plots.extend(_plot_categorical_frequencies(structured_df, cat_cols, plots_dir))

    outlier_flags_df, outlier_summary = _compute_outlier_flags(structured_df, id_col=id_col, numeric_cols=numeric_cols)
    outlier_flags_path = reports_dir / "outlier_flags.parquet"
    outlier_flags_df.to_parquet(outlier_flags_path, index=False)

    numeric_desc = structured_df[numeric_cols].describe().transpose() if numeric_cols else pd.DataFrame()
    numeric_stats = (
        numeric_desc[["mean", "std", "min", "max"]].round(4).to_dict(orient="index")
        if not numeric_desc.empty
        else {}
    )

    categorical_top_values = {
        col: structured_df[col].astype(str).value_counts(dropna=False).head(10).to_dict() for col in cat_cols
    }

    summary_payload = {
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "missing_ratio": {col: float(structured_df[col].isna().mean()) for col in structured_df.columns},
        "numeric_stats": numeric_stats,
        "categorical_top_values": categorical_top_values,
        "outlier_summary": outlier_summary,
        "generated_plots": generated_plots,
    }

    eda_summary_path = reports_dir / "eda_summary.json"
    write_json(eda_summary_path, summary_payload)

    return EdaResult(
        eda_summary_path=eda_summary_path,
        outlier_flags_path=outlier_flags_path,
        generated_plots=generated_plots,
    )
