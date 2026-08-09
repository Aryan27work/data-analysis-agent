from __future__ import annotations

import pytest

from agent.state import InsightItem, InsightList
from utils.llm_client import _normalize_structured_result


def test_structured_result_dict_is_normalized():

    data = {
        "insights": [
            {
                "claim": "Average sales are 100.",
                "interpretation": None,
                "recommendation": None,
                "statistic": "mean",
                "source_column": "sales",
                "comparison_column": None,
                "category_value": None,
                "claimed_value": 100.0,
                "confidence": "high",
            }
        ]
    }

    result = _normalize_structured_result(
        data,
        InsightList,
    )

    assert isinstance(
        result,
        InsightList,
    )

    assert isinstance(
        result.insights,
        list,
    )

    assert isinstance(
        result.insights[0],
        InsightItem,
    )


def test_structured_result_never_returns_raw_list():

    data = {
        "insights": []
    }

    result = _normalize_structured_result(
        data,
        InsightList,
    )

    assert isinstance(
        result,
        InsightList,
    )

    assert not isinstance(
        result,
        list,
    )


def test_invalid_structured_result_is_rejected():

    with pytest.raises(
        ValueError
    ):

        _normalize_structured_result(
            ["this", "is", "wrong"],
            InsightList,
        )