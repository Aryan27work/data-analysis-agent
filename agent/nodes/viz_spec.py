from __future__ import annotations

import json
from typing import Any

from agent.prompts import VIZ_SPEC_SYSTEM
from agent.state import ChartSpecList
from utils.chart_validation import filter_valid_specs
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


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _build_fallback_chart_specs(
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a small deterministic set of safe charts when the LLM produces
    invalid or unusable chart specifications.

    This function does NOT call an LLM.

    The fallback is deliberately conservative:
      1. Prefer a numeric histogram.
      2. Add a numeric box plot if another numeric column exists.
      3. Add a categorical bar chart if a categorical column exists.
      4. Add a scatter plot when at least two numeric columns exist.
      5. Add a correlation heatmap when at least two numeric columns exist.

    Every generated spec is validated through the same deterministic
    chart-validation layer before being returned.
    """

    columns_meta = profile.get("columns", {})

    if not isinstance(columns_meta, dict):
        return []

    continuous_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []

    for column, metadata in columns_meta.items():
        if not isinstance(metadata, dict):
            continue

        cardinality_class = metadata.get(
            "cardinality_class"
        )

        if cardinality_class == "continuous":
            continuous_columns.append(column)

        elif cardinality_class == "categorical":
            categorical_columns.append(column)

        elif cardinality_class == "datetime":
            datetime_columns.append(column)

    candidates: list[dict[str, Any]] = []

    # 1. Histogram
    if continuous_columns:
        candidates.append(
            {
                "type": "histogram",
                "columns": [continuous_columns[0]],
            }
        )

    # 2. Box plot
    if len(continuous_columns) >= 2:
        candidates.append(
            {
                "type": "box",
                "columns": [continuous_columns[1]],
            }
        )
    elif continuous_columns:
        candidates.append(
            {
                "type": "box",
                "columns": [continuous_columns[0]],
            }
        )

    # 3. Categorical bar chart
    if categorical_columns:
        candidates.append(
            {
                "type": "bar",
                "columns": [categorical_columns[0]],
            }
        )

    # 4. Scatter plot
    if len(continuous_columns) >= 2:
        candidates.append(
            {
                "type": "scatter",
                "columns": [
                    continuous_columns[0],
                    continuous_columns[1],
                ],
            }
        )

    # 5. Correlation heatmap
    if len(continuous_columns) >= 2:
        candidates.append(
            {
                "type": "correlation_heatmap",
                "columns": continuous_columns[:5],
            }
        )

    # Optional line chart when a datetime column and numeric column exist.
    if datetime_columns and continuous_columns:
        candidates.append(
            {
                "type": "line",
                "columns": [
                    datetime_columns[0],
                    continuous_columns[0],
                ],
            }
        )

    # Validate fallback specs through the exact same deterministic guardrail.
    valid, _ = filter_valid_specs(
        candidates,
        profile,
    )

    # The renderer only needs a few useful charts.
    return valid[:5]


# ---------------------------------------------------------------------------
# LLM proposal validation
# ---------------------------------------------------------------------------

def _validate_chart_specs(
    specs: ChartSpecList,
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Convert the structured LLM response into dictionaries and apply the
    deterministic chart validation layer.

    Returns:
        (valid_specs, rejection_reasons)
    """

    if not isinstance(specs, ChartSpecList):
        return [], ["LLM did not return a valid ChartSpecList."]

    raw_specs: list[dict[str, Any]] = []

    for spec in specs.charts:
        try:
            data = spec.model_dump()
        except Exception as exc:
            logger.warning(
                "chart_spec_model_dump_failed error=%s",
                exc.__class__.__name__,
            )
            continue

        if isinstance(data, dict):
            raw_specs.append(data)

    valid_specs, rejection_reasons = filter_valid_specs(
        raw_specs,
        profile,
    )

    return valid_specs, rejection_reasons


# ---------------------------------------------------------------------------
# Visualization specification node
# ---------------------------------------------------------------------------

@timed_node
def visualization_spec_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate chart specifications.

    Primary path:
        LLM -> structured ChartSpecList -> deterministic validation

    Fallback path:
        If all LLM proposals are rejected, generate deterministic chart
        specifications from the dataset profile.

    No invalid chart specification is allowed to reach the renderer.
    """

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    profile = state.get(
        "profile",
        {},
    )

    stats = state.get(
        "stats",
        {},
    )

    if not isinstance(profile, dict):
        profile = {}

    if not isinstance(stats, dict):
        stats = {}

    try:
        # ---------------------------------------------------------------
        # Build compact context for the LLM
        # ---------------------------------------------------------------

        context, warnings = build_llm_context(
            profile,
            stats,
        )

        errors.extend(warnings)

        insights = state.get(
            "insights",
            [],
        )

        if not isinstance(insights, list):
            insights = []

        # ---------------------------------------------------------------
        # Ask LLM for chart specifications
        # ---------------------------------------------------------------

        prompt = f"""
{VIZ_SPEC_SYSTEM}

Choose 3-5 useful visualizations based only on
the supplied dataset profile, statistics and insights.

Dataset context:
{json.dumps(context, default=str, indent=2)}

Insights:
{json.dumps(insights, default=str, indent=2)}

Return chart specifications only.
""".strip()

        result = structured_call(
            prompt,
            ChartSpecList,
        )

        # ---------------------------------------------------------------
        # Deterministically validate LLM proposals
        # ---------------------------------------------------------------

        chart_specs, rejection_reasons = _validate_chart_specs(
            result,
            profile,
        )

        # Preserve rejection information so it is visible to the user.
        errors.extend(rejection_reasons)

        # ---------------------------------------------------------------
        # Deterministic fallback
        # ---------------------------------------------------------------

        if not chart_specs:
            errors.append(
                "LLM produced no valid chart specifications. "
                "Using deterministic fallback charts."
            )

            chart_specs = _build_fallback_chart_specs(
                profile,
            )

        # ---------------------------------------------------------------
        # Final safety check
        #
        # This is intentionally repeated after fallback generation.
        # Nothing should reach chart_render.py unless it passes the same
        # deterministic validation rules.
        # ---------------------------------------------------------------

        chart_specs, final_rejections = filter_valid_specs(
            chart_specs,
            profile,
        )

        errors.extend(final_rejections)

        if not chart_specs:
            errors.append(
                "No valid chart specifications could be generated, "
                "including deterministic fallback charts."
            )

        return {
            "chart_specs": chart_specs,
            "errors": errors,
        }

    except (
        LLMCallError,
        LLMConfigError,
        ValueError,
        TypeError,
    ) as exc:

        logger.exception(
            "visualization_spec_failed",
        )

        errors.append(
            f"Visualization specification failed: {exc}"
        )

        # Even if the LLM itself fails, try deterministic charts.
        fallback_specs = _build_fallback_chart_specs(
            profile,
        )

        if fallback_specs:
            errors.append(
                "Using deterministic fallback charts after "
                "visualization specification failure."
            )

        return {
            "chart_specs": fallback_specs,
            "errors": errors,
        }