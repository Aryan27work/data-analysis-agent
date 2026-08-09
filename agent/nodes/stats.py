"""Pure-Python statistical analysis node.

No LLM call is made here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.config import MAX_DETAILED_COLUMNS
from utils.logging_config import get_logger, timed_node


logger = get_logger(__name__)


MAX_GROUP_CATEGORICAL_COLS = 5
MAX_GROUP_NUMERIC_COLS = 5
MAX_GROUP_CARDINALITY = 15


def _iqr_outlier_count(
    series: pd.Series,
) -> int:

    clean = series.dropna()

    if clean.empty:
        return 0

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(
        (
            (clean < lower)
            | (clean > upper)
        ).sum()
    )


def _numeric_column_stats(
    series: pd.Series,
) -> dict[str, Any]:

    desc = series.describe().to_dict()

    return {
        "count": desc.get("count"),
        "mean": desc.get("mean"),
        "std": desc.get("std"),
        "min": desc.get("min"),
        "25%": desc.get("25%"),
        "50%": desc.get("50%"),
        "75%": desc.get("75%"),
        "max": desc.get("max"),
        "skew": (
            float(series.skew())
            if series.notna().sum() > 2
            else None
        ),
        "kurtosis": (
            float(series.kurt())
            if series.notna().sum() > 3
            else None
        ),
        "outlier_count_iqr": _iqr_outlier_count(
            series
        ),
    }


def _categorical_column_stats(
    series: pd.Series,
) -> dict[str, Any]:

    non_null = series.dropna()

    vc = non_null.value_counts()

    return {
        "value_counts": {
            str(k): int(v)
            for k, v in vc.head(10).items()
        },
        "cardinality": int(
            series.nunique(
                dropna=True
            )
        ),
        "total_count": int(
            non_null.shape[0]
        ),
    }


def _compute_group_stats(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> dict[str, dict[str, dict[str, float]]]:

    group_stats: dict[
        str,
        dict[str, dict[str, float]],
    ] = {}

    eligible_cat = categorical_cols[
        :MAX_GROUP_CATEGORICAL_COLS
    ]

    eligible_num = numeric_cols[
        :MAX_GROUP_NUMERIC_COLS
    ]

    for cat_col in eligible_cat:

        if (
            df[cat_col].nunique(dropna=True)
            > MAX_GROUP_CARDINALITY
        ):
            continue

        for num_col in eligible_num:

            try:

                means = (
                    df.groupby(
                        cat_col,
                        observed=True,
                    )[num_col]
                    .mean()
                    .round(3)
                )

                group_stats.setdefault(
                    cat_col,
                    {},
                )[num_col] = {
                    str(key): float(value)
                    for key, value in means.items()
                    if pd.notna(value)
                }

            except Exception as exc:

                logger.warning(
                    "group_stats_failed "
                    "cat_col=%s num_col=%s error=%s",
                    cat_col,
                    num_col,
                    exc.__class__.__name__,
                )

    return group_stats


@timed_node
def stats_node(
    state: dict,
) -> dict:

    df: pd.DataFrame = state["df"]

    profile: dict[str, Any] = state[
        "profile"
    ]

    errors: list[str] = list(
        state.get(
            "errors",
            [],
        )
    )

    columns_meta = profile.get(
        "columns",
        {},
    )

    numeric_cols = [
        column
        for column, metadata
        in columns_meta.items()
        if metadata.get(
            "cardinality_class"
        ) == "continuous"
    ]

    categorical_cols = [
        column
        for column, metadata
        in columns_meta.items()
        if metadata.get(
            "cardinality_class"
        ) == "categorical"
    ]

    datetime_cols = profile.get(
        "datetime_columns",
        [],
    )

    total_detailed = (
        len(numeric_cols)
        + len(categorical_cols)
    )

    if total_detailed > MAX_DETAILED_COLUMNS:

        numeric_limit = (
            MAX_DETAILED_COLUMNS // 2
        )

        categorical_limit = (
            MAX_DETAILED_COLUMNS
            - numeric_limit
        )

        keep_numeric = numeric_cols[
            :numeric_limit
        ]

        keep_categorical = categorical_cols[
            :categorical_limit
        ]

        dropped = (
            set(numeric_cols)
            - set(keep_numeric)
        ) | (
            set(categorical_cols)
            - set(keep_categorical)
        )

        if dropped:

            errors.append(
                "Dataset is very wide; detailed stats "
                f"limited to {MAX_DETAILED_COLUMNS} "
                f"columns. Skipped: {sorted(dropped)}"
            )

        numeric_cols = keep_numeric
        categorical_cols = keep_categorical

    numeric_stats = {
        column: _numeric_column_stats(
            df[column]
        )
        for column in numeric_cols
    }

    categorical_stats = {
        column: _categorical_column_stats(
            df[column]
        )
        for column in categorical_cols
    }

    correlations: dict[str, Any] = {}

    if len(numeric_cols) >= 2:

        corr_matrix = (
            df[numeric_cols]
            .corr(
                numeric_only=True
            )
            .round(3)
        )

        correlations = corr_matrix.to_dict()

    else:

        errors.append(
            "Fewer than 2 numeric columns; "
            "correlation matrix skipped."
        )

    group_stats: dict[str, Any] = {}

    if categorical_cols and numeric_cols:

        group_stats = _compute_group_stats(
            df,
            categorical_cols,
            numeric_cols,
        )

    datetime_trend: dict[str, Any] = {}

    if datetime_cols and numeric_cols:

        dt_col = datetime_cols[0]
        num_col = numeric_cols[0]

        try:

            parsed = pd.to_datetime(
                df[dt_col],
                errors="coerce",
                format="mixed",
            )

            temp = pd.DataFrame(
                {
                    dt_col: parsed,
                    num_col: df[num_col],
                }
            ).dropna()

            if not temp.empty:

                monthly = (
                    temp.set_index(dt_col)[num_col]
                    .resample("MS")
                    .agg(
                        ["count", "mean"]
                    )
                    .round(3)
                )

                datetime_trend = {
                    "column": dt_col,
                    "measured_column": num_col,
                    "monthly": {
                        str(index.date()): row.to_dict()
                        for index, row
                        in monthly.iterrows()
                    },
                }

        except Exception as exc:

            errors.append(
                f"Could not compute datetime trend "
                f"for '{dt_col}': {exc}"
            )

    stats = {
        "numeric": numeric_stats,
        "categorical": categorical_stats,
        "correlations": correlations,
        "group_stats": group_stats,
        "datetime_trend": datetime_trend,
    }

    return {
        "stats": stats,
        "errors": errors,
    }