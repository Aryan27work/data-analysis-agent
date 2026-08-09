"""End-to-end smoke tests for the compiled graph, using a mocked LLM client
so CI doesn't need a real Gemini API key / network access.

Covers:
- the graph compiles and reaches END with every state key populated
- the conditional retry loop cannot exceed MAX_RETRIES regardless of what
  the LLM proposes
- the (now deterministic, Python-only) evidence system catches a
  deliberately-induced bad insight — the number it cites simply isn't the
  real mean in the dataset — and the pipeline still terminates and flags it
- chart spec validation rejects a hallucinated column/type before rendering
"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.state import InsightList, InsightItem, ChartSpecList, ChartSpec
from agent.graph import build_graph, MAX_RETRIES

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _sample_df():
    return pd.read_csv(os.path.join(FIXTURES, "numeric_heavy.csv"))


GOOD_INSIGHTS = InsightList(insights=[
    InsightItem(
        claim="The dataset contains 300 records.",
        statistic="count",
        source_column="amount",
        claimed_value=300.0,
        confidence="high",
    ),
])

# The real mean(amount) in numeric_heavy.csv is ~65, not 9999.99 — this is a
# deliberately fabricated claim to prove the evidence system catches it.
BAD_INSIGHTS = InsightList(insights=[
    InsightItem(
        claim="Average transaction amount is an extreme $9,999.99.",
        statistic="mean",
        source_column="amount",
        claimed_value=9999.99,
        confidence="high",
    ),
])

CHART_SPECS = ChartSpecList(charts=[
    ChartSpec(type="histogram", columns=["amount"], reason="test"),
])

# Deliberately invalid: histogram requires exactly one numeric column.
INVALID_CHART_SPECS = ChartSpecList(charts=[
    ChartSpec(type="histogram", columns=["amount", "distance_from_home"], reason="bad spec"),
    ChartSpec(type="histogram", columns=["nonexistent_column"], reason="hallucinated column"),
])


def _mock_always_good(prompt, schema, temperature=0.2):
    if schema.__name__ == "InsightList":
        return GOOD_INSIGHTS
    if schema.__name__ == "ChartSpecList":
        return CHART_SPECS
    raise ValueError(f"Unexpected schema {schema}")


def _mock_bad_insight_never_fixed(prompt, schema, temperature=0.2):
    """Always returns the deliberately bad insight, regardless of retry —
    used to test that retries are capped and the pipeline still terminates."""
    if schema.__name__ == "InsightList":
        return BAD_INSIGHTS
    if schema.__name__ == "ChartSpecList":
        return CHART_SPECS
    raise ValueError(f"Unexpected schema {schema}")


def _mock_invalid_charts(prompt, schema, temperature=0.2):
    if schema.__name__ == "InsightList":
        return GOOD_INSIGHTS
    if schema.__name__ == "ChartSpecList":
        return INVALID_CHART_SPECS
    raise ValueError(f"Unexpected schema {schema}")


@patch("agent.nodes.report.plain_call", return_value="# Report\nMock report body.")
@patch("agent.nodes.viz_spec.structured_call", side_effect=_mock_always_good)
@patch("agent.nodes.insights.structured_call", side_effect=_mock_always_good)
def test_graph_runs_end_to_end(mock_ins, mock_viz, mock_report):
    graph = build_graph()
    df = _sample_df()
    final_state = graph.invoke({"file_path": "test.csv", "df": df, "retry_count": 0, "errors": []})

    assert final_state["insights_verified"] is True
    assert final_state["final_report"].startswith("# Report")
    assert "Statistical Evidence" in final_state["final_report"]
    assert len(final_state["chart_paths"]) >= 1
    assert final_state["retry_count"] == 1  # verified on the first attempt


@patch("agent.nodes.report.plain_call", return_value="# Report\nMock report body.")
@patch("agent.nodes.viz_spec.structured_call", side_effect=_mock_bad_insight_never_fixed)
@patch("agent.nodes.insights.structured_call", side_effect=_mock_bad_insight_never_fixed)
def test_retry_loop_terminates_and_flags_bad_insight(mock_ins, mock_viz, mock_report):
    """The deliberately-induced-bad-insight test referenced in the
    deliverable checklist. The evidence system (agent/evidence.py) must
    reject the fabricated $9,999.99 mean every time by direct comparison
    against the real computed mean, the loop must retry up to MAX_RETRIES,
    and the pipeline must still terminate and flag the insight as
    lower-confidence rather than looping forever or silently accepting it."""
    graph = build_graph()
    df = _sample_df()
    final_state = graph.invoke({"file_path": "test.csv", "df": df, "retry_count": 0, "errors": []})

    # Loop must terminate exactly at MAX_RETRIES verification attempts.
    assert final_state["retry_count"] == MAX_RETRIES

    # The bad insight must never pass verification, and the evidence record
    # must show the real value it was checked against.
    assert final_state["insights_verified"] is False
    assert final_state["per_insight_passed"] == [False]
    evidence = final_state["evidence"][0]
    assert evidence["passed"] is False
    assert evidence["claimed_value"] == 9999.99
    assert evidence["actual_value"] is not None
    assert evidence["actual_value"] != 9999.99

    # Pipeline still reaches the end and produces a report despite the
    # unresolved verification failure, with the evidence table included.
    assert final_state["final_report"].startswith("# Report")
    assert "❌" in final_state["final_report"]


@patch("agent.nodes.report.plain_call", return_value="# Report\nMock report body.")
@patch("agent.nodes.viz_spec.structured_call", side_effect=_mock_invalid_charts)
@patch("agent.nodes.insights.structured_call", side_effect=_mock_always_good)
def test_invalid_chart_specs_are_rejected_not_crashed(mock_ins, mock_viz, mock_report):
    """A hallucinated column and a wrong-column-count chart spec must both
    be filtered out by utils.chart_validation before rendering, with the
    pipeline falling back to a deterministic chart set rather than crashing."""
    graph = build_graph()
    df = _sample_df()
    final_state = graph.invoke({"file_path": "test.csv", "df": df, "retry_count": 0, "errors": []})

    assert len(final_state["chart_paths"]) >= 1  # fallback charts still produced
    assert any("Rejected chart spec" in e or "fallback" in e for e in final_state["errors"])
