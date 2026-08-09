"""Deterministic validation for LLM-proposed chart specs. Nothing here calls
an LLM — this is the guardrail that stops a hallucinated chart type, wrong
column count, or incompatible dtype from ever reaching the renderer.

Used in two places by design (belt and suspenders): visualization_spec_node
filters LLM proposals through this before accepting them, and
chart_rendering_node re-checks defensively right before rendering in case
state was constructed some other way (e.g. a future node, or a test).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

# Column-count and cardinality-class rules per chart type.
CHART_RULES: Dict[str, Dict[str, Any]] = {
    "histogram": {"min_cols": 1, "max_cols": 1, "required_classes": [{"continuous"}]},
    "box": {"min_cols": 1, "max_cols": 1, "required_classes": [{"continuous"}]},
    "scatter": {"min_cols": 2, "max_cols": 2, "required_classes": [{"continuous"}, {"continuous"}]},
    "bar": {"min_cols": 1, "max_cols": 2, "required_classes": [{"categorical"}, {"continuous"}]},
    "line": {"min_cols": 2, "max_cols": 2, "required_classes": [{"datetime", "categorical"}, {"continuous"}]},
    "correlation_heatmap": {"min_cols": 2, "max_cols": None, "required_classes": [{"continuous"}]},
}


def validate_chart_spec(
    spec: Dict[str, Any], profile: Dict[str, Any], df: pd.DataFrame | None = None
) -> Tuple[bool, str]:
    """Returns (is_valid, reason). reason explains the rejection when invalid,
    or is empty when valid."""
    chart_type = spec.get("type")
    columns = spec.get("columns", [])

    rules = CHART_RULES.get(chart_type)
    if rules is None:
        return False, f"Unknown chart type '{chart_type}'."

    if not isinstance(columns, list) or not columns:
        return False, "Chart spec has no columns."

    n = len(columns)
    if n < rules["min_cols"] or (rules["max_cols"] is not None and n > rules["max_cols"]):
        expected = (
            f"{rules['min_cols']}" if rules["min_cols"] == rules["max_cols"]
            else f"{rules['min_cols']}-{rules['max_cols'] or 'N'}"
        )
        return False, f"'{chart_type}' requires {expected} column(s), got {n}."

    columns_meta = profile.get("columns", {})
    for col in columns:
        if col not in columns_meta:
            return False, f"Column '{col}' does not exist in the dataset."

    # correlation_heatmap and bar (2nd col) have flexible-length or optional
    # slots; check dtype compatibility per-position where a fixed rule exists.
    if chart_type == "correlation_heatmap":
        bad = [c for c in columns if columns_meta[c]["cardinality_class"] != "continuous"]
        if bad:
            return False, f"'correlation_heatmap' requires numeric columns; got non-numeric {bad}."
    elif chart_type == "bar":
        cat_col = columns[0]
        if columns_meta[cat_col]["cardinality_class"] not in ("categorical",):
            return False, f"'bar' requires a categorical first column; '{cat_col}' is {columns_meta[cat_col]['cardinality_class']}."
        if n == 2:
            num_col = columns[1]
            if columns_meta[num_col]["cardinality_class"] != "continuous":
                return False, f"'bar' second column must be numeric; '{num_col}' is {columns_meta[num_col]['cardinality_class']}."
    else:
        for col, allowed_classes in zip(columns, rules["required_classes"]):
            actual_class = columns_meta[col]["cardinality_class"]
            if actual_class not in allowed_classes:
                return False, (
                    f"'{chart_type}' requires column(s) of type {allowed_classes}; "
                    f"'{col}' is '{actual_class}'."
                )

    if df is not None:
        for col in columns:
            if col in df.columns and df[col].notna().sum() == 0:
                return False, f"Column '{col}' has no non-null values to plot."

    return True, ""


def filter_valid_specs(
    specs: List[Dict[str, Any]], profile: Dict[str, Any], df: pd.DataFrame | None = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validates a list of chart specs, returning (valid_specs, rejection_reasons)."""
    valid = []
    reasons = []
    for spec in specs:
        ok, reason = validate_chart_spec(spec, profile, df)
        if ok:
            valid.append(spec)
        else:
            reasons.append(f"Rejected chart spec {spec}: {reason}")
    return valid, reasons
