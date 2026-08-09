"""Deterministic evidence verification.

No LLM calls occur in this module.

Every LLM-generated claimed_value is checked against the actual statistics
computed by Python.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.state import Evidence, InsightItem


RELATIVE_TOLERANCE = 0.02
ABSOLUTE_TOLERANCE = 0.01


NUMERIC_STAT_KEY_MAP = {
    "mean": "mean",
    "median": "50%",
    "std": "std",
    "min": "min",
    "max": "max",
    "skew": "skew",
    "outlier_count": "outlier_count_iqr",
}


def _values_match(
    claimed: float,
    actual: float,
) -> bool:

    tolerance = max(
        RELATIVE_TOLERANCE * abs(actual),
        ABSOLUTE_TOLERANCE,
    )

    return abs(
        claimed - actual
    ) <= tolerance


def _lookup_actual_value(
    profile: dict[str, Any],
    stats: dict[str, Any],
    insight: InsightItem,
) -> Optional[float]:

    stat = insight.statistic
    col = insight.source_column
    comp = insight.comparison_column
    cat_val = insight.category_value

    numeric = stats.get(
        "numeric",
        {},
    )

    categorical = stats.get(
        "categorical",
        {},
    )

    correlations = stats.get(
        "correlations",
        {},
    )

    group_stats = stats.get(
        "group_stats",
        {},
    )

    datetime_trend = stats.get(
        "datetime_trend",
        {},
    )

    if stat in NUMERIC_STAT_KEY_MAP:

        col_stats = numeric.get(
            col
        )

        if not col_stats:
            return None

        value = col_stats.get(
            NUMERIC_STAT_KEY_MAP[stat]
        )

        if value is None:
            return None

        return float(value)

    if stat == "count":

        if cat_val is not None:

            cat_stats = categorical.get(
                col
            )

            if not cat_stats:
                return None

            value = cat_stats.get(
                "value_counts",
                {},
            ).get(cat_val)

            if value is None:
                return None

            return float(value)

        shape = profile.get(
            "shape",
            {},
        )

        rows = shape.get(
            "rows"
        )

        return (
            float(rows)
            if rows is not None
            else None
        )

    if stat == "percentage":

        cat_stats = categorical.get(
            col
        )

        if not cat_stats or cat_val is None:
            return None

        count = cat_stats.get(
            "value_counts",
            {},
        ).get(cat_val)

        total = cat_stats.get(
            "total_count"
        )

        if count is None or not total:
            return None

        return round(
            100 * count / total,
            3,
        )

    if stat == "null_percentage":

        col_meta = (
            profile
            .get("columns", {})
            .get(col)
        )

        if not col_meta:
            return None

        value = col_meta.get(
            "null_pct"
        )

        return (
            float(value)
            if value is not None
            else None
        )

    if stat == "cardinality":

        cat_stats = categorical.get(
            col
        )

        if cat_stats:

            value = cat_stats.get(
                "cardinality"
            )

            if value is not None:
                return float(value)

        col_meta = (
            profile
            .get("columns", {})
            .get(col)
        )

        if not col_meta:
            return None

        value = col_meta.get(
            "unique_count"
        )

        return (
            float(value)
            if value is not None
            else None
        )

    if stat == "correlation":

        if comp is None:
            return None

        row = correlations.get(
            col,
            {},
        )

        value = row.get(
            comp
        )

        if value is None:

            row = correlations.get(
                comp,
                {},
            )

            value = row.get(
                col
            )

        return (
            float(value)
            if value is not None
            else None
        )

    if stat == "group_mean":

        if (
            comp is None
            or cat_val is None
        ):
            return None

        value = (
            group_stats
            .get(col, {})
            .get(comp, {})
            .get(cat_val)
        )

        return (
            float(value)
            if value is not None
            else None
        )

    if stat == "trend":

        monthly = datetime_trend.get(
            "monthly",
            {},
        )

        if cat_val is None:
            return None

        entry = monthly.get(
            cat_val
        )

        if not entry:
            return None

        value = (
            entry.get("mean")
            if comp
            else entry.get("count")
        )

        return (
            float(value)
            if value is not None
            else None
        )

    return None


def evaluate_insight(
    profile: dict[str, Any],
    stats: dict[str, Any],
    insight: InsightItem,
) -> Evidence:

    actual = _lookup_actual_value(
        profile,
        stats,
        insight,
    )

    if actual is None:

        return Evidence(
            claim=insight.claim,
            statistic=insight.statistic,
            source_column=insight.source_column,
            comparison_column=insight.comparison_column,
            category_value=insight.category_value,
            claimed_value=insight.claimed_value,
            actual_value=None,
            passed=False,
            reason=(
                f"Could not locate a computed "
                f"'{insight.statistic}' statistic for "
                f"column '{insight.source_column}'"
                + (
                    f" / '{insight.comparison_column}'"
                    if insight.comparison_column
                    else ""
                )
                + (
                    f" / group '{insight.category_value}'"
                    if insight.category_value
                    else ""
                )
                + " — this claim is unverifiable "
                "against the actual data."
            ),
        )

    if _values_match(
        insight.claimed_value,
        actual,
    ):

        return Evidence(
            claim=insight.claim,
            statistic=insight.statistic,
            source_column=insight.source_column,
            comparison_column=insight.comparison_column,
            category_value=insight.category_value,
            claimed_value=insight.claimed_value,
            actual_value=actual,
            passed=True,
            reason=(
                "Claimed value matches the "
                "computed statistic."
            ),
        )

    return Evidence(
        claim=insight.claim,
        statistic=insight.statistic,
        source_column=insight.source_column,
        comparison_column=insight.comparison_column,
        category_value=insight.category_value,
        claimed_value=insight.claimed_value,
        actual_value=actual,
        passed=False,
        reason=(
            f"Claimed {insight.claimed_value} but "
            f"the actual computed {insight.statistic} "
            f"is {actual}."
        ),
    )


def evaluate_all(
    profile: dict[str, Any],
    stats: dict[str, Any],
    insights: list[dict[str, Any]],
) -> list[Evidence]:

    evidences: list[Evidence] = []

    for raw in insights:

        try:

            insight = InsightItem.model_validate(
                raw
            )

        except Exception as exc:

            evidences.append(
                Evidence(
                    claim=str(
                        raw.get(
                            "claim",
                            "",
                        )
                    ),
                    statistic=str(
                        raw.get(
                            "statistic",
                            "unknown",
                        )
                    ),
                    source_column=str(
                        raw.get(
                            "source_column",
                            "unknown",
                        )
                    ),
                    claimed_value=raw.get(
                        "claimed_value"
                    ),
                    actual_value=None,
                    passed=False,
                    reason=(
                        "Insight did not match the "
                        f"expected schema: {exc}"
                    ),
                )
            )

            continue

        evidences.append(
            evaluate_insight(
                profile,
                stats,
                insight,
            )
        )

    return evidences