from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes.chart_render import chart_rendering_node
from agent.nodes.insights import insight_generation_node
from agent.nodes.report import report_synthesis_node
from agent.nodes.stats import stats_node
from agent.nodes.verification import insight_verification_node
from agent.nodes.viz_spec import visualization_spec_node
from agent.nodes.profiler import profiler_node
from agent.state import AgentState


MAX_RETRIES = 2


def _route_after_verification(
    state: AgentState,
) -> str:
    """
    Deterministic routing after insight verification.

    verified
        -> visualization

    failed + retries remaining
        -> regenerate insights

    failed + retries exhausted
        -> visualization with warnings
    """

    verified = bool(
        state.get(
            "insights_verified",
            False,
        )
    )

    if verified:
        return "visualization"

    retry_count = int(
        state.get(
            "retry_count",
            0,
        )
    )

    if retry_count < MAX_RETRIES:
        return "retry"

    return "visualization"


def build_graph():
    """
    Build and compile the complete analysis graph.
    """

    graph = StateGraph(
        AgentState,
    )

    graph.add_node(
        "profiler",
        profiler_node,
    )

    graph.add_node(
        "stats",
        stats_node,
    )

    graph.add_node(
        "insights",
        insight_generation_node,
    )

    graph.add_node(
        "verification",
        insight_verification_node,
    )

    graph.add_node(
        "visualization",
        visualization_spec_node,
    )

    graph.add_node(
        "chart_render",
        chart_rendering_node,
    )

    graph.add_node(
        "report",
        report_synthesis_node,
    )

    graph.add_edge(
        START,
        "profiler",
    )

    graph.add_edge(
        "profiler",
        "stats",
    )

    graph.add_edge(
        "stats",
        "insights",
    )

    graph.add_edge(
        "insights",
        "verification",
    )

    graph.add_conditional_edges(
        "verification",
        _route_after_verification,
        {
            "retry": "insights",
            "visualization": "visualization",
        },
    )

    graph.add_edge(
        "visualization",
        "chart_render",
    )

    graph.add_edge(
        "chart_render",
        "report",
    )

    graph.add_edge(
        "report",
        END,
    )

    return graph.compile()