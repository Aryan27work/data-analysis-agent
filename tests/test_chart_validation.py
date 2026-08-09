import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.chart_validation import validate_chart_spec, filter_valid_specs

PROFILE = {
    "columns": {
        "amount": {"cardinality_class": "continuous"},
        "distance": {"cardinality_class": "continuous"},
        "age": {"cardinality_class": "continuous"},
        "category": {"cardinality_class": "categorical"},
        "order_date": {"cardinality_class": "datetime"},
        "notes": {"cardinality_class": "text"},
    }
}


def test_valid_histogram():
    ok, reason = validate_chart_spec({"type": "histogram", "columns": ["amount"]}, PROFILE)
    assert ok, reason


def test_histogram_wrong_column_count():
    ok, reason = validate_chart_spec(
        {"type": "histogram", "columns": ["amount", "distance"]}, PROFILE
    )
    assert not ok
    assert "requires" in reason


def test_histogram_wrong_dtype():
    ok, reason = validate_chart_spec({"type": "histogram", "columns": ["category"]}, PROFILE)
    assert not ok


def test_valid_scatter():
    ok, reason = validate_chart_spec(
        {"type": "scatter", "columns": ["amount", "distance"]}, PROFILE
    )
    assert ok, reason


def test_scatter_needs_two_numeric_columns():
    ok, reason = validate_chart_spec(
        {"type": "scatter", "columns": ["amount", "category"]}, PROFILE
    )
    assert not ok


def test_valid_bar_single_categorical():
    ok, reason = validate_chart_spec({"type": "bar", "columns": ["category"]}, PROFILE)
    assert ok, reason


def test_valid_bar_grouped_aggregate():
    ok, reason = validate_chart_spec(
        {"type": "bar", "columns": ["category", "amount"]}, PROFILE
    )
    assert ok, reason


def test_bar_rejects_numeric_first_column():
    ok, reason = validate_chart_spec({"type": "bar", "columns": ["amount"]}, PROFILE)
    assert not ok


def test_valid_line():
    ok, reason = validate_chart_spec(
        {"type": "line", "columns": ["order_date", "amount"]}, PROFILE
    )
    assert ok, reason


def test_valid_correlation_heatmap():
    ok, reason = validate_chart_spec(
        {"type": "correlation_heatmap", "columns": ["amount", "distance", "age"]}, PROFILE
    )
    assert ok, reason


def test_correlation_heatmap_needs_at_least_two_columns():
    ok, reason = validate_chart_spec(
        {"type": "correlation_heatmap", "columns": ["amount"]}, PROFILE
    )
    assert not ok


def test_unknown_chart_type_rejected():
    ok, reason = validate_chart_spec({"type": "pie_of_lies", "columns": ["amount"]}, PROFILE)
    assert not ok
    assert "Unknown chart type" in reason


def test_missing_column_rejected():
    ok, reason = validate_chart_spec({"type": "histogram", "columns": ["does_not_exist"]}, PROFILE)
    assert not ok
    assert "does not exist" in reason


def test_filter_valid_specs_separates_good_and_bad():
    specs = [
        {"type": "histogram", "columns": ["amount"]},
        {"type": "histogram", "columns": ["category"]},  # invalid dtype
        {"type": "bar", "columns": ["category"]},
        {"type": "made_up_type", "columns": ["amount"]},  # invalid type
    ]
    valid, reasons = filter_valid_specs(specs, PROFILE)
    assert len(valid) == 2
    assert len(reasons) == 2
