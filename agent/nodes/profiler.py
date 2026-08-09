"""profiler_node — pure Python, no LLM call.

Populates state["profile"] with shape, per-column dtype/null/cardinality info,
a guess at the target variable, detected datetime columns, and PII flags.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from utils.logging_config import timed_node

TARGET_NAME_HINTS = {"target", "label", "flag", "outcome", "class", "y", "churn", "fraud"}
PII_NAME_HINTS = {"name", "email", "phone", "address", "ssn", "dob", "birth", "zip",
                   "postal", "passport", "credit_card", "card_number"}

MAX_ID_UNIQUENESS_RATIO = 0.98
DATETIME_PARSE_SUCCESS_THRESHOLD = 0.9


def _is_stringlike(series: pd.Series) -> bool:
    """True for classic object-dtype strings AND pandas' newer dedicated
    string dtype (pandas >= 2.x with the string backend, or 3.x default)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _looks_like_id_name(col_name: str) -> bool:
    lowered = col_name.lower()
    return lowered == "id" or lowered.endswith("_id") or lowered.startswith("id_")


def _classify_column(series: pd.Series, col_name: str, n_rows: int) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    n_unique = series.nunique(dropna=True)
    uniqueness_ratio = n_unique / n_rows if n_rows else 0

    if pd.api.types.is_numeric_dtype(series):
        is_integer_like = pd.api.types.is_integer_dtype(series)
        # Only call it an ID column if it's near-unique AND either integer-typed
        # or explicitly named like an identifier — a near-unique *float* column
        # (e.g. transaction amounts) is continuous data, not an ID.
        if uniqueness_ratio > MAX_ID_UNIQUENESS_RATIO and n_unique > 50 and (
            is_integer_like or _looks_like_id_name(col_name)
        ):
            return "id"
        # Low-cardinality numeric often behaves like a categorical flag
        if n_unique <= 10:
            return "categorical"
        return "continuous"

    if _is_stringlike(series):
        if uniqueness_ratio > MAX_ID_UNIQUENESS_RATIO:
            return "id"
        if n_unique <= max(20, int(0.05 * n_rows)):
            return "categorical"
        return "text"

    return "other"


def _looks_like_pii(col_name: str) -> bool:
    lowered = col_name.lower()
    return any(hint in lowered for hint in PII_NAME_HINTS)


def _try_parse_datetime(series: pd.Series) -> bool:
    if not _is_stringlike(series):
        return False
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    sample = sample.sample(min(200, len(sample)), random_state=0)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    success_rate = parsed.notna().mean()
    return success_rate >= DATETIME_PARSE_SUCCESS_THRESHOLD


def _detect_target(profile_columns: Dict[str, Dict[str, Any]], n_rows: int) -> str | None:
    # Prefer name-based hints first.
    for col, meta in profile_columns.items():
        if any(hint in col.lower() for hint in TARGET_NAME_HINTS):
            return col
    # Fall back to a binary/low-cardinality categorical column.
    candidates = [
        col for col, meta in profile_columns.items()
        if meta["cardinality_class"] == "categorical" and meta["unique_count"] <= 5
    ]
    return candidates[0] if candidates else None


@timed_node
def profiler_node(state: dict) -> dict:
    df: pd.DataFrame = state["df"]
    errors: List[str] = list(state.get("errors", []))

    n_rows, n_cols = df.shape
    columns_profile: Dict[str, Dict[str, Any]] = {}
    datetime_columns: List[str] = []
    pii_columns: List[str] = []
    all_null_columns: List[str] = []

    for col in df.columns:
        series = df[col]
        null_pct = round(series.isna().mean() * 100, 2)
        unique_count = int(series.nunique(dropna=True))

        if null_pct >= 100.0:
            # Entirely empty column — already flagged by utils.validation,
            # but the profiler is the source of truth other nodes read from,
            # so it must also exclude these from stats/insight generation.
            cardinality_class = "all_null"
            all_null_columns.append(col)
        else:
            is_datetime_candidate = (
                _try_parse_datetime(series) if _is_stringlike(series) else False
            )
            if is_datetime_candidate:
                datetime_columns.append(col)
            cardinality_class = (
                "datetime" if is_datetime_candidate else _classify_column(series, col, n_rows)
            )

        col_meta = {
            "dtype": str(series.dtype),
            "null_pct": null_pct,
            "unique_count": unique_count,
            "cardinality_class": cardinality_class,
        }
        columns_profile[col] = col_meta

        if _looks_like_pii(col):
            pii_columns.append(col)

    likely_target = _detect_target(columns_profile, n_rows)

    if n_rows < 2:
        errors.append("Dataset has fewer than 2 rows; statistics and insights will be minimal.")
    if n_cols < 2:
        errors.append("Dataset has only 1 column; correlation and multi-variable analysis are skipped.")

    profile = {
        "shape": {"rows": n_rows, "columns": n_cols},
        "columns": columns_profile,
        "likely_target": likely_target,
        "datetime_columns": datetime_columns,
        "likely_pii_columns": pii_columns,
        "all_null_columns": all_null_columns,
    }

    return {"profile": profile, "errors": errors}
