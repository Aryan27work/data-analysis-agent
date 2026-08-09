import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.nodes.profiler import profiler_node
from agent.nodes.stats import stats_node

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    return pd.read_csv(os.path.join(FIXTURES, name))


def _profile_and_stats(name):
    df = _load(name)
    state = {"df": df, "errors": []}
    state.update(profiler_node(state))
    state.update(stats_node(state))
    return state


def test_numeric_heavy_stats():
    state = _profile_and_stats("numeric_heavy.csv")
    stats = state["stats"]
    assert "amount" in stats["numeric"]
    assert "outlier_count_iqr" in stats["numeric"]["amount"]
    assert stats["correlations"]  # 2+ numeric cols present


def test_categorical_heavy_stats():
    state = _profile_and_stats("categorical_heavy.csv")
    stats = state["stats"]
    assert "plan_type" in stats["categorical"]
    assert "value_counts" in stats["categorical"]["plan_type"]


def test_datetime_trend_computed():
    state = _profile_and_stats("datetime_mixed.csv")
    stats = state["stats"]
    assert stats["datetime_trend"].get("column") == "order_date"
    assert stats["datetime_trend"].get("monthly")


def test_one_row_stats_do_not_crash():
    state = _profile_and_stats("one_row.csv")
    assert state["stats"] is not None


def test_one_column_correlation_skipped():
    state = _profile_and_stats("one_column.csv")
    assert state["stats"]["correlations"] == {}
    assert any("Fewer than 2 numeric" in e for e in state["errors"])
