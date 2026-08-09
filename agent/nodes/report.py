"""report_synthesis_node — LLM call, plus a deterministic Python-generated
evidence table appended afterward.

The LLM writes the narrative sections (executive summary, key findings in
prose, business interpretation, recommendations). The "Statistical Evidence"
table showing claimed vs. actual values is built directly from
state["evidence"] in Python and appended verbatim — the LLM cannot alter or
omit it, which is what "the LLM should never be able to bypass verification"
means in the final artifact, not just internally.
"""

from __future__ import annotations

import json
import os
from typing import List

from agent.prompts import REPORT_SYNTHESIS_SYSTEM
from utils.context_budget import build_llm_context
from utils.llm_client import plain_call, LLMCallError, LLMConfigError
from utils.logging_config import get_logger, timed_node

logger = get_logger(__name__)


def _evidence_table(evidence: List[dict]) -> str:
    if not evidence:
        return ""
    lines = [
        "## Statistical Evidence",
        "",
        "_Generated directly from computed statistics — not written by the LLM._",
        "",
        "| # | Claim | Statistic | Column | Claimed | Actual | Verified |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(evidence, 1):
        claim = (e.get("claim") or "").replace("|", "\\|")
        col = e.get("source_column", "")
        if e.get("comparison_column"):
            col = f"{col} / {e['comparison_column']}"
        claimed = e.get("claimed_value")
        actual = e.get("actual_value")
        verified = "✅" if e.get("passed") else "❌"
        lines.append(
            f"| {i} | {claim} | {e.get('statistic', '')} | {col} | "
            f"{claimed if claimed is not None else '—'} | "
            f"{actual if actual is not None else 'not found'} | {verified} |"
        )
    return "\n".join(lines)


def _fallback_report(state: dict) -> str:
    profile = state["profile"]
    insights = state.get("insights", [])
    errors = state.get("errors", [])
    lines = [
        "# Data Analysis Report (fallback — LLM synthesis unavailable)",
        "",
        "## Dataset Overview",
        f"- Rows: {profile['shape']['rows']}, Columns: {profile['shape']['columns']}",
        "",
        "## Key Findings",
    ]
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. {ins.get('claim', '')} (statistic: {ins.get('statistic', 'n/a')} = {ins.get('claimed_value', 'n/a')})")
    if errors:
        lines.append("\n## Data Quality Notes")
        for e in errors:
            lines.append(f"- {e}")
    return "\n".join(lines)


@timed_node
def report_synthesis_node(state: dict) -> dict:
    errors: List[str] = list(state.get("errors", []))
    profile = state["profile"]
    stats = state["stats"]
    insights = state.get("insights", [])
    chart_paths = state.get("chart_paths", [])
    evidence = state.get("evidence", [])
    insights_verified = state.get("insights_verified", True)
    retry_count = state.get("retry_count", 0)

    context, budget_warnings = build_llm_context(profile, stats)
    errors.extend(budget_warnings)

    chart_names = [os.path.basename(p) for p in chart_paths]

    low_confidence_note = ""
    if not insights_verified:
        failed_claims = [e["claim"] for e in evidence if not e.get("passed")]
        low_confidence_note = (
            f"\n\nNote: verification did not fully pass after {retry_count} attempt(s). "
            f"These insights are lower-confidence and must be clearly flagged, not "
            f"presented as reliable findings: {failed_claims}"
        )

    prompt = (
        f"{REPORT_SYNTHESIS_SYSTEM}\n\n"
        f"Dataset statistics (JSON):\n{json.dumps(context, default=str)}\n\n"
        f"Insights, each already tagged fact/interpretation/recommendation "
        f"(JSON):\n{json.dumps(insights, default=str)}\n\n"
        f"Chart filenames available (reference these, do not invent others): {chart_names}\n\n"
        f"Data quality warnings collected during the pipeline: {errors}"
        f"{low_confidence_note}"
    )

    try:
        report_body = plain_call(prompt)
    except (LLMCallError, LLMConfigError) as exc:
        errors.append(f"Report synthesis failed, using fallback report: {exc}")
        logger.error("report_synthesis_failed error=%s", exc)
        report_body = _fallback_report(state)

    evidence_section = _evidence_table(evidence)
    final_report = report_body
    if evidence_section:
        final_report = f"{report_body}\n\n{evidence_section}"

    return {"final_report": final_report, "errors": errors}
