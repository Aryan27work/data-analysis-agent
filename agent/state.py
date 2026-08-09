from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


StatisticType = Literal[
    "mean",
    "median",
    "std",
    "min",
    "max",
    "skew",
    "outlier_count",
    "count",
    "percentage",
    "null_percentage",
    "cardinality",
    "correlation",
    "group_mean",
    "trend",
]


class InsightItem(BaseModel):
    """
    A single traceable analytical insight.

    Every numerical claim must point to a deterministic statistic that
    Python calculated before the LLM saw it.
    """

    model_config = ConfigDict(extra="ignore")

    claim: str = Field(
        ...,
        min_length=1,
        description="A factual statement supported by the cited statistic.",
    )

    interpretation: Optional[str] = Field(
        default=None,
        description="Optional interpretation without introducing new numbers.",
    )

    recommendation: Optional[str] = Field(
        default=None,
        description="Optional action based on the finding.",
    )

    statistic: StatisticType = Field(
        ...,
        description="Statistic type calculated by Python.",
    )

    source_column: str = Field(
        ...,
        min_length=1,
        description="Exact source column name.",
    )

    comparison_column: Optional[str] = Field(
        default=None,
        description="Second column for correlation/group statistics.",
    )

    category_value: Optional[str] = Field(
        default=None,
        description="Category/month/group when applicable.",
    )

    claimed_value: float = Field(
        ...,
        description="Numerical value claimed by the insight.",
    )

    confidence: Literal["high", "medium", "low"] = "medium"


class InsightList(BaseModel):
    """
    Root structured response from the insight-generation LLM call.
    """

    model_config = ConfigDict(extra="ignore")

    insights: List[InsightItem] = Field(
        default_factory=list,
    )


class ChartSpec(BaseModel):
    """
    One visualization requested by the LLM.
    """

    model_config = ConfigDict(extra="ignore")

    type: Literal[
        "histogram",
        "bar",
        "scatter",
        "correlation_heatmap",
        "line",
        "box",
    ]

    columns: List[str] = Field(
        ...,
        min_length=1,
    )

    reason: str = Field(
        ...,
        min_length=1,
    )


class ChartSpecList(BaseModel):
    """
    Root structured response from the chart-selection LLM call.
    """

    model_config = ConfigDict(extra="ignore")

    charts: List[ChartSpec] = Field(
        default_factory=list,
    )


class Evidence(BaseModel):
    """
    Deterministic Python verification result.

    The LLM never controls `passed`.
    """

    model_config = ConfigDict(extra="ignore")

    claim: str

    statistic: str

    source_column: str

    comparison_column: Optional[str] = None

    category_value: Optional[str] = None

    claimed_value: Optional[float] = None

    actual_value: Optional[float] = None

    passed: bool

    reason: str


class AgentState(TypedDict, total=False):
    file_path: str

    df: pd.DataFrame

    profile: Dict[str, Any]

    stats: Dict[str, Any]

    # IMPORTANT:
    # Graph state always stores serialized dictionaries.
    insights: List[Dict[str, Any]]

    evidence: List[Dict[str, Any]]

    insights_verified: bool

    verification_feedback: str

    per_insight_passed: List[bool]

    chart_specs: List[Dict[str, Any]]

    chart_paths: List[str]

    final_report: str

    retry_count: int

    errors: List[str]