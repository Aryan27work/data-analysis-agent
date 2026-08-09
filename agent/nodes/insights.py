from __future__ import annotations

import json
from typing import Any

from agent.prompts import (
    INSIGHT_GENERATION_SYSTEM,
    INSIGHT_RETRY_PREFIX,
)
from agent.state import InsightList
from utils.context_budget import build_llm_context
from utils.llm_client import (
    LLMCallError,
    LLMConfigError,
    structured_call,
)
from utils.logging_config import (
    get_logger,
    timed_node,
)


logger = get_logger(__name__)


def _column_reference_block(
    profile: dict[str, Any],
) -> str:
    """
    Build a compact description of valid columns/statistics.
    """

    lines: list[str] = []

    columns = profile.get("columns", {})

    if not isinstance(columns, dict):
        return "No column metadata available."

    for column_name, metadata in columns.items():

        if not isinstance(metadata, dict):
            continue

        cardinality_class = metadata.get(
            "cardinality_class",
            "unknown",
        )

        if cardinality_class == "continuous":

            hint = (
                "mean, median, std, min, max, "
                "skew, outlier_count, correlation, group_mean"
            )

        elif cardinality_class == "categorical":

            hint = (
                "count, percentage, cardinality, group_mean"
            )

        elif cardinality_class == "datetime":

            hint = "trend"

        else:

            hint = (
                "count, null_percentage, cardinality"
            )

        lines.append(
            f"- {column_name} "
            f"({cardinality_class}): {hint}"
        )

    return "\n".join(lines)


def _serialize_insights(
    result: InsightList,
) -> list[dict[str, Any]]:
    """
    Convert InsightList into the exact graph-state format.

    This is the critical boundary:

        Pydantic model
            ↓
        List[dict]
            ↓
        AgentState

    Never return the Pydantic root list directly.
    """

    if not isinstance(result, InsightList):
        raise TypeError(
            "Expected InsightList, received "
            f"{type(result).__name__}"
        )

    if not isinstance(result.insights, list):
        raise TypeError(
            "InsightList.insights must be a list."
        )

    output: list[dict[str, Any]] = []

    for item in result.insights:

        data = item.model_dump(
            mode="python",
        )

        if not isinstance(data, dict):
            raise TypeError(
                "Serialized InsightItem must be a dictionary."
            )

        output.append(data)

    return output


@timed_node
def insight_generation_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    errors = list(
        state.get("errors", [])
    )

    profile = state.get("profile")

    stats = state.get("stats")

    if not isinstance(profile, dict):
        errors.append(
            "Insight generation skipped: invalid profile."
        )

        return {
            "insights": [],
            "errors": errors,
        }

    if not isinstance(stats, dict):
        errors.append(
            "Insight generation skipped: invalid statistics."
        )

        return {
            "insights": [],
            "errors": errors,
        }

    try:

        context, warnings = build_llm_context(
            profile,
            stats,
        )

        errors.extend(warnings)

        prompt_parts = [
            INSIGHT_GENERATION_SYSTEM,
        ]

        feedback = state.get(
            "verification_feedback",
        )

        if feedback:
            prompt_parts.append(
                INSIGHT_RETRY_PREFIX.format(
                    feedback=feedback,
                )
            )

        prompt_parts.append(
            "Available columns and valid "
            "statistic types:\n"
            + _column_reference_block(profile)
        )

        prompt_parts.append(
            "Dataset profile and statistics "
            "(JSON):\n"
            + json.dumps(
                context,
                default=str,
                ensure_ascii=False,
            )
        )

        prompt = "\n\n".join(
            prompt_parts,
        )

        result = structured_call(
            prompt,
            InsightList,
        )

        insights = _serialize_insights(
            result,
        )

        if not insights:
            raise ValueError(
                "LLM returned zero insights."
            )

        return {
            "insights": insights,
            "errors": errors,
        }

    except (
        LLMCallError,
        LLMConfigError,
        ValueError,
        TypeError,
    ) as exc:

        logger.exception(
            "insight_generation_failed",
        )

        errors.append(
            f"Insight generation failed: {exc}"
        )

        # Preserve previous valid insights during retries
        # instead of destroying them.
        previous = state.get(
            "insights",
            [],
        )

        if not isinstance(previous, list):
            previous = []

        if previous:
            return {
                "insights": previous,
                "errors": errors,
            }

        columns = profile.get(
            "columns",
            {},
        )

        if not isinstance(columns, dict):
            columns = {}

        first_column = (
            next(iter(columns), "unknown")
        )

        rows = (
            profile
            .get("shape", {})
            .get("rows", 0)
        )

        fallback = {
            "claim": (
                "Insight generation is unavailable "
                "because the LLM call failed."
            ),
            "interpretation": None,
            "recommendation": None,
            "statistic": "count",
            "source_column": first_column,
            "comparison_column": None,
            "category_value": None,
            "claimed_value": float(rows or 0),
            "confidence": "low",
        }

        return {
            "insights": [fallback],
            "errors": errors,
        }