"""Build compact, token-budgeted context for LLM calls.

Raw pandas-derived structures can become very large. This module creates a
bounded JSON-friendly representation before sending dataset information to
the LLM.

PII-flagged columns never have their raw values included.
"""

from __future__ import annotations

import json
from typing import Any

from utils.config import CONTEXT_TOKEN_CEILING


CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_CEILING = CONTEXT_TOKEN_CEILING


def _round_floats(
    obj: Any,
    ndigits: int = 3,
) -> Any:

    if isinstance(obj, float):
        return round(obj, ndigits)

    if isinstance(obj, dict):
        return {
            key: _round_floats(value, ndigits)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            _round_floats(value, ndigits)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return [
            _round_floats(value, ndigits)
            for value in obj
        ]

    return obj


def _truncate_value_counts(
    col_stats: dict[str, Any],
    top_n: int = 10,
) -> dict[str, Any]:

    vc = col_stats.get("value_counts")

    if isinstance(vc, dict) and len(vc) > top_n:

        items = list(vc.items())

        kept = dict(
            items[:top_n]
        )

        other_total = sum(
            value
            for _, value in items[top_n:]
        )

        if other_total:
            kept["__other__"] = other_total

        col_stats = {
            **col_stats,
            "value_counts": kept,
        }

    return col_stats


def _strip_pii(
    profile: dict[str, Any],
    stats: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:

    pii_cols = set(
        profile.get(
            "likely_pii_columns",
            [],
        )
    )

    if not pii_cols:
        return profile, stats

    profile = dict(profile)
    stats = dict(stats)

    numeric = dict(
        stats.get(
            "numeric",
            {},
        )
    )

    categorical = dict(
        stats.get(
            "categorical",
            {},
        )
    )

    group_stats = dict(
        stats.get(
            "group_stats",
            {},
        )
    )

    for col in pii_cols:

        if col in numeric:
            numeric[col] = {
                "note": (
                    "PII-flagged column; "
                    "values withheld from LLM."
                )
            }

        if col in categorical:
            categorical[col] = {
                "note": (
                    "PII-flagged column; "
                    "values withheld from LLM."
                )
            }

        group_stats.pop(
            col,
            None,
        )

    stats["numeric"] = numeric
    stats["categorical"] = categorical
    stats["group_stats"] = group_stats

    return profile, stats


def estimate_tokens(obj: Any) -> int:
    """Estimate token count using a lightweight character heuristic."""

    serialized = json.dumps(
        obj,
        default=str,
    )

    return max(
        1,
        len(serialized) // CHARS_PER_TOKEN,
    )


def build_llm_context(
    profile: dict[str, Any],
    stats: dict[str, Any],
    token_ceiling: int = DEFAULT_TOKEN_CEILING,
) -> tuple[dict[str, Any], list[str]]:

    warnings: list[str] = []

    profile, stats = _strip_pii(
        profile,
        stats,
    )

    numeric = {
        col: _round_floats(stat)
        for col, stat in stats.get(
            "numeric",
            {},
        ).items()
    }

    categorical = {
        col: _truncate_value_counts(
            _round_floats(stat)
        )
        for col, stat in stats.get(
            "categorical",
            {},
        ).items()
    }

    context = {
        "shape": profile.get("shape"),
        "columns": profile.get("columns"),
        "likely_target": profile.get("likely_target"),
        "datetime_columns": profile.get(
            "datetime_columns",
            [],
        ),
        "numeric_stats": numeric,
        "categorical_stats": categorical,
        "correlations": _round_floats(
            stats.get(
                "correlations",
                {},
            )
        ),
        "group_stats": _round_floats(
            stats.get(
                "group_stats",
                {},
            )
        ),
        "datetime_trend": _round_floats(
            stats.get(
                "datetime_trend",
                {},
            )
        ),
    }

    dropped: list[str] = []

    if (
        estimate_tokens(context) > token_ceiling
        and context["group_stats"]
    ):
        context["group_stats"] = {}

        dropped.append(
            "group_stats (all)"
        )

    def _cardinality(
        col_name: str,
    ) -> int:

        stats_for_column = categorical.get(
            col_name,
            {},
        )

        return int(
            stats_for_column.get(
                "cardinality",
                0,
            )
            or 0
        )

    sorted_cat_cols = sorted(
        categorical.keys(),
        key=_cardinality,
        reverse=True,
    )

    index = 0

    while (
        estimate_tokens(context) > token_ceiling
        and index < len(sorted_cat_cols)
    ):

        col = sorted_cat_cols[index]

        if col in context["categorical_stats"]:

            del context["categorical_stats"][col]

            dropped.append(col)

        index += 1

    if dropped:
        warnings.append(
            f"Context budget exceeded {token_ceiling} tokens; "
            "dropped detailed statistics before sending to LLM: "
            f"{dropped}"
        )

    return context, warnings