from __future__ import annotations

from typing import Any

from agent.evidence import evaluate_all
from utils.logging_config import (
    get_logger,
    timed_node,
)

logger = get_logger(__name__)

MAX_RETRIES = 2


def _normalize_insights(
    value: Any,
) -> list[dict[str, Any]]:
    """
    Normalize graph-state insights into dictionaries.
    """

    if value is None:
        return []

    if not isinstance(value, list):
        raise TypeError(
            "state['insights'] must be a list, "
            f"received {type(value).__name__}"
        )

    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(value):

        if isinstance(item, dict):
            normalized.append(item)
            continue

        if hasattr(item, "model_dump"):
            dumped = item.model_dump()

            if isinstance(dumped, dict):
                normalized.append(dumped)
                continue

        raise TypeError(
            f"Insight at index {index} must be a dict, "
            f"received {type(item).__name__}"
        )

    return normalized


@timed_node
def insight_verification_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    errors = list(state.get("errors", []))

    current_retry_count = int(
        state.get("retry_count", 0)
    )

    # IMPORTANT:
    # retry_count represents verification attempts.
    # Therefore every verification execution increments it,
    # including a successful verification.
    attempt_number = min(
        current_retry_count + 1,
        MAX_RETRIES,
    )

    try:

        insights = _normalize_insights(
            state.get("insights", [])
        )

        stats = state.get(
            "stats",
            {},
        )

        profile = state.get(
            "profile",
            {},
        )

        if not isinstance(stats, dict):
            raise TypeError(
                "state['stats'] must be a dictionary."
            )

        if not isinstance(profile, dict):
            raise TypeError(
                "state['profile'] must be a dictionary."
            )

        if not insights:
            errors.append(
                "No insights available for verification."
            )

            return {
                "insights": insights,
                "insights_verified": False,
                "verification_feedback": (
                    "No insights were generated."
                ),
                "per_insight_passed": [],
                "evidence": [],
                "retry_count": attempt_number,
                "errors": errors,
            }

        # evaluate_all signature:
        #
        # evaluate_all(
        #     profile,
        #     stats,
        #     insights,
        # )
        #
        # This order is important.
        evidence = evaluate_all(
            profile,
            stats,
            insights,
        )

        if not isinstance(evidence, list):
            raise TypeError(
                "evaluate_all() must return a list."
            )

        normalized_evidence: list[dict[str, Any]] = []

        for index, item in enumerate(evidence):

            if hasattr(item, "model_dump"):
                item = item.model_dump()

            if not isinstance(item, dict):
                raise TypeError(
                    "Evidence at index "
                    f"{index} is not a dictionary."
                )

            normalized_evidence.append(item)

        passed = [
            bool(
                item.get(
                    "passed",
                    False,
                )
            )
            for item in normalized_evidence
        ]

        verified = (
            bool(passed)
            and all(passed)
        )

        failed_reasons = [
            str(
                item.get(
                    "reason",
                    "Evidence verification failed.",
                )
            )
            for item in normalized_evidence
            if not item.get("passed", False)
        ]

        if verified:

            feedback = ""

        else:

            if failed_reasons:
                feedback = (
                    "The following insights failed "
                    "deterministic evidence verification:\n"
                    + "\n".join(
                        f"- {reason}"
                        for reason in failed_reasons
                    )
                )

            else:
                feedback = (
                    "One or more insights failed "
                    "deterministic evidence verification."
                )

        return {
            "insights": insights,
            "evidence": normalized_evidence,
            "insights_verified": verified,
            "verification_feedback": feedback,
            "per_insight_passed": passed,
            "retry_count": attempt_number,
            "errors": errors,
        }

    except Exception as exc:

        logger.exception(
            "verification_failed"
        )

        errors.append(
            f"Insight verification failed: {exc}"
        )

        return {
            "insights_verified": False,
            "verification_feedback": (
                "Verification failed because the "
                "verification system encountered an error."
            ),
            "per_insight_passed": [],
            "evidence": [],
            "retry_count": attempt_number,
            "errors": errors,
        }