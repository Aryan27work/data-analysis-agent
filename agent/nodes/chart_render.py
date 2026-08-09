"""chart_rendering_node — pure Python, no LLM call.

Renders each chart spec with Matplotlib/Seaborn into an ephemeral temp
directory (required for Streamlit Cloud's read-only/ephemeral filesystem
outside of tempfile). Every spec is re-validated defensively (belt and
suspenders on top of visualization_spec_node's validation) and every render
is wrapped in try/except so one bad spec can't crash the pipeline.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless rendering, required for server environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.chart_validation import validate_chart_spec
from utils.logging_config import get_logger, timed_node

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def _render_histogram(df: pd.DataFrame, columns: List[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df[columns[0]].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {columns[0]}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_bar(df: pd.DataFrame, columns: List[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(columns) == 2:
        # categorical + numeric -> grouped aggregate (mean) bar chart
        cat_col, num_col = columns
        means = df.groupby(cat_col, observed=True)[num_col].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=means.values, y=means.index.astype(str), ax=ax)
        ax.set_title(f"Mean {num_col} by {cat_col}")
        ax.set_xlabel(f"Mean {num_col}")
    else:
        counts = df[columns[0]].value_counts(dropna=True).head(10)
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax)
        ax.set_title(f"Top categories: {columns[0]}")
        ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_scatter(df: pd.DataFrame, columns: List[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(x=df[columns[0]], y=df[columns[1]], ax=ax, alpha=0.6)
    ax.set_title(f"{columns[0]} vs {columns[1]}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_correlation_heatmap(df: pd.DataFrame, columns: List[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[columns].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_line(df: pd.DataFrame, columns: List[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_sorted = df.sort_values(columns[0])
    ax.plot(df_sorted[columns[0]], df_sorted[columns[1]])
    ax.set_title(f"Trend: {columns[1]} over {columns[0]}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_box(df: pd.DataFrame, columns: List[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(x=df[columns[0]], ax=ax)
    ax.set_title(f"Box plot: {columns[0]}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


RENDERERS = {
    "histogram": _render_histogram,
    "bar": _render_bar,
    "scatter": _render_scatter,
    "correlation_heatmap": _render_correlation_heatmap,
    "line": _render_line,
    "box": _render_box,
}


@timed_node
def chart_rendering_node(state: dict) -> dict:
    df: pd.DataFrame = state["df"]
    profile = state["profile"]
    specs = state.get("chart_specs", [])
    errors: List[str] = list(state.get("errors", []))

    out_dir = tempfile.mkdtemp(prefix="agent_charts_")
    chart_paths: List[str] = []

    for i, spec in enumerate(specs):
        # Defensive re-validation, in case chart_specs was populated by
        # something other than visualization_spec_node (e.g. a future node,
        # or a test) that skipped the earlier validation step.
        ok, reason = validate_chart_spec(spec, profile, df)
        if not ok:
            errors.append(f"Skipped chart {i+1}: {reason}")
            continue

        chart_type = spec["type"]
        columns = spec["columns"]
        renderer = RENDERERS[chart_type]
        out_path = os.path.join(out_dir, f"chart_{i+1}_{chart_type}.png")
        try:
            renderer(df, columns, out_path)
            chart_paths.append(out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chart_render_failed type=%s error=%s", chart_type, exc.__class__.__name__)
            errors.append(
                f"Failed to render chart {i+1} ({chart_type}, columns={columns}): {exc}"
            )
            continue

    return {"chart_paths": chart_paths, "errors": errors}
