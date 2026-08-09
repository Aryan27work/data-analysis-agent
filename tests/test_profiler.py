import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.nodes.profiler import profiler_node

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    return pd.read_csv(os.path.join(FIXTURES, name))


def test_numeric_heavy_profile():
    df = _load("numeric_heavy.csv")
    out = profiler_node({"df": df, "errors": []})
    profile = out["profile"]
    assert profile["shape"]["rows"] == 300
    assert "amount" in profile["columns"]
    assert profile["columns"]["transaction_id"]["cardinality_class"] == "id"
    assert profile["likely_target"] == "fraud_flag"


def test_categorical_heavy_profile():
    df = _load("categorical_heavy.csv")
    out = profiler_node({"df": df, "errors": []})
    profile = out["profile"]
    assert profile["columns"]["plan_type"]["cardinality_class"] == "categorical"
    assert profile["likely_target"] in ("churn", "plan_type", "satisfaction")


def test_datetime_detection():
    df = _load("datetime_mixed.csv")
    out = profiler_node({"df": df, "errors": []})
    profile = out["profile"]
    assert "order_date" in profile["datetime_columns"]


def test_one_row_does_not_crash():
    df = _load("one_row.csv")
    out = profiler_node({"df": df, "errors": []})
    assert out["profile"]["shape"]["rows"] == 1
    assert any("fewer than 2 rows" in e for e in out["errors"])


def test_one_column_does_not_crash():
    df = _load("one_column.csv")
    out = profiler_node({"df": df, "errors": []})
    assert out["profile"]["shape"]["columns"] == 1
    assert any("only 1 column" in e for e in out["errors"])


def test_all_text_dataset():
    df = _load("all_text.csv")
    out = profiler_node({"df": df, "errors": []})
    profile = out["profile"]
    # should not crash and should classify the free-text column sensibly
    assert profile["columns"]["notes"]["cardinality_class"] in ("text", "categorical", "id")
