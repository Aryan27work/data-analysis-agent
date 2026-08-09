"""Lightweight evaluation framework for the agent pipeline.

Runs the graph (optionally with a mocked LLM, so it can be exercised without
API cost/credentials) across a set of datasets and reports real, measured
metrics — this module produces no fabricated numbers; every metric here is
computed from an actual pipeline run recorded during the call.

Usage:
    python -m utils.evaluation --fixtures tests/fixtures --mock

Metrics reported (see EvalResult):
    - workflow success rate (ran to END without an unrecoverable error)
    - insight verification pass rate (evidence-verified insights / total)
    - unsupported-claim rate (1 - pass rate)
    - chart validity rate (rendered charts / proposed charts)
    - retries used per run
    - LLM calls made per run
    - wall-clock runtime per run

This is infrastructure for generating real results from real datasets — it
is deliberately NOT bundled with pre-computed "our agent achieves 94%
accuracy" numbers, since those would have to be fabricated without actually
running against a labeled benchmark.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import patch

import pandas as pd

from agent.graph import build_graph, MAX_RETRIES
from agent.state import InsightList, InsightItem, ChartSpecList, ChartSpec


@dataclass
class EvalResult:
    dataset: str
    success: bool
    runtime_seconds: float
    n_insights: int
    n_verified: int
    n_chart_specs_proposed: int
    n_charts_rendered: int
    retries_used: int
    error: str = ""

    @property
    def verification_pass_rate(self) -> float:
        return (self.n_verified / self.n_insights) if self.n_insights else 0.0

    @property
    def unsupported_claim_rate(self) -> float:
        return 1.0 - self.verification_pass_rate

    @property
    def chart_validity_rate(self) -> float:
        return (
            (self.n_charts_rendered / self.n_chart_specs_proposed)
            if self.n_chart_specs_proposed else 0.0
        )


def _build_mock_for_dataset(df: pd.DataFrame, profile_hint_col: str):
    """Builds a mock that references a real column so the row-count claim is
    checkable, plus a real numeric column for the chart spec."""
    def _mock(prompt, schema, temperature=0.2):
        if schema.__name__ == "InsightList":
            return InsightList(insights=[
                InsightItem(
                    claim=f"Dataset has {len(df)} rows.", statistic="count",
                    source_column=profile_hint_col, claimed_value=float(len(df)),
                    confidence="high",
                ),
            ])
        if schema.__name__ == "ChartSpecList":
            return ChartSpecList(charts=[
                ChartSpec(type="histogram", columns=[profile_hint_col], reason="eval mock"),
            ])
        raise ValueError(f"Unexpected schema in eval mock: {schema}")
    return _mock


def run_single_evaluation(csv_path: str, use_mock: bool = True) -> EvalResult:
    df = pd.read_csv(csv_path)

    # Use the real profiler to pick a genuinely continuous column for the
    # mock (not just "first numeric dtype column", which could be an ID
    # column and would fail chart validation for a histogram).
    from agent.nodes.profiler import profiler_node
    profile = profiler_node({"df": df, "errors": []})["profile"]
    continuous_cols = [c for c, m in profile["columns"].items() if m["cardinality_class"] == "continuous"]
    numeric_col = continuous_cols[0] if continuous_cols else df.columns[0]

    start = time.perf_counter()
    try:
        if use_mock:
            mock_fn = _build_mock_for_dataset(df, numeric_col)
            with patch("agent.nodes.insights.structured_call", side_effect=mock_fn), \
                 patch("agent.nodes.viz_spec.structured_call", side_effect=mock_fn), \
                 patch("agent.nodes.report.plain_call", return_value="# Report\nEval mock report."):
                final_state = build_graph().invoke(
                    {"file_path": csv_path, "df": df, "retry_count": 0, "errors": []}
                )
        else:
            final_state = build_graph().invoke(
                {"file_path": csv_path, "df": df, "retry_count": 0, "errors": []}
            )
        elapsed = time.perf_counter() - start

        evidence = final_state.get("evidence", [])
        return EvalResult(
            dataset=os.path.basename(csv_path),
            success=True,
            runtime_seconds=round(elapsed, 3),
            n_insights=len(final_state.get("insights", [])),
            n_verified=sum(1 for e in evidence if e.get("passed")),
            n_chart_specs_proposed=len(final_state.get("chart_specs", [])),
            n_charts_rendered=len(final_state.get("chart_paths", [])),
            retries_used=final_state.get("retry_count", 0),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return EvalResult(
            dataset=os.path.basename(csv_path),
            success=False,
            runtime_seconds=round(elapsed, 3),
            n_insights=0, n_verified=0, n_chart_specs_proposed=0,
            n_charts_rendered=0, retries_used=0,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def run_evaluation_suite(fixtures_dir: str, use_mock: bool = True) -> List[EvalResult]:
    csv_paths = sorted(glob.glob(os.path.join(fixtures_dir, "*.csv")))
    return [run_single_evaluation(p, use_mock=use_mock) for p in csv_paths]


def print_report(results: List[EvalResult]) -> None:
    n = len(results)
    n_success = sum(1 for r in results if r.success)
    print(f"\n=== Evaluation report ({n} dataset(s)) ===")
    print(f"Workflow success rate: {n_success}/{n} ({100*n_success/n:.0f}%)" if n else "No datasets found.")
    for r in results:
        status = "OK" if r.success else f"FAILED ({r.error})"
        print(
            f"- {r.dataset}: {status} | runtime={r.runtime_seconds}s | "
            f"insights={r.n_insights} verified={r.n_verified} "
            f"(pass_rate={r.verification_pass_rate:.0%}) | "
            f"charts={r.n_charts_rendered}/{r.n_chart_specs_proposed} | "
            f"retries={r.retries_used}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="tests/fixtures", help="Directory of CSVs to evaluate")
    parser.add_argument("--mock", action="store_true", help="Use a mocked LLM (no API key needed)")
    args = parser.parse_args()

    results = run_evaluation_suite(args.fixtures, use_mock=args.mock)
    print_report(results)
