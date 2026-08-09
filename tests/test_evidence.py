import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.evidence import evaluate_insight, evaluate_all
from agent.state import InsightItem

PROFILE = {
    "shape": {"rows": 300, "columns": 4},
    "columns": {
        "amount": {"dtype": "float64", "null_pct": 0.0, "unique_count": 300, "cardinality_class": "continuous"},
        "fraud_flag": {"dtype": "int64", "null_pct": 0.0, "unique_count": 2, "cardinality_class": "categorical"},
    },
}

STATS = {
    "numeric": {
        "amount": {
            "count": 300.0, "mean": 65.135, "std": 12.4, "min": 10.0,
            "25%": 40.0, "50%": 55.0, "75%": 70.0, "max": 520.0,
            "skew": 1.8, "kurtosis": 5.2, "outlier_count_iqr": 10,
        },
        "distance_from_home": {
            "count": 300.0, "mean": 20.0, "std": 5.0, "min": 0.0,
            "25%": 10.0, "50%": 18.0, "75%": 25.0, "max": 60.0,
            "skew": 0.5, "kurtosis": 1.0, "outlier_count_iqr": 3,
        },
    },
    "categorical": {
        "fraud_flag": {
            "value_counts": {"0": 287, "1": 13},
            "cardinality": 2,
            "total_count": 300,
        },
    },
    "correlations": {
        "amount": {"amount": 1.0, "distance_from_home": 0.03},
        "distance_from_home": {"amount": 0.03, "distance_from_home": 1.0},
    },
    "group_stats": {
        "fraud_flag": {
            "amount": {"0": 65.939, "1": 47.386},
        },
    },
    "datetime_trend": {
        "column": "order_date", "measured_column": "amount",
        "monthly": {
            "2024-01-01": {"count": 30, "mean": 62.5},
        },
    },
}


def test_mean_claim_passes_when_correct():
    insight = InsightItem(
        claim="Average amount is 65.14.",
        statistic="mean", source_column="amount", claimed_value=65.135,
    )
    evidence = evaluate_insight(PROFILE, STATS, insight)
    assert evidence.passed
    assert evidence.actual_value == 65.135


def test_mean_claim_fails_when_wrong():
    insight = InsightItem(
        claim="Average amount is 9999.99.",
        statistic="mean", source_column="amount", claimed_value=9999.99,
    )
    evidence = evaluate_insight(PROFILE, STATS, insight)
    assert not evidence.passed
    assert "9999.99" in evidence.reason


def test_mean_claim_within_rounding_tolerance_passes():
    insight = InsightItem(
        claim="Average amount is about 65.13.",
        statistic="mean", source_column="amount", claimed_value=65.13,
    )
    evidence = evaluate_insight(PROFILE, STATS, insight)
    assert evidence.passed


def test_percentage_claim():
    insight = InsightItem(
        claim="4.33% of transactions are flagged as fraud.",
        statistic="percentage", source_column="fraud_flag",
        category_value="1", claimed_value=round(100 * 13 / 300, 3),
    )
    evidence = evaluate_insight(PROFILE, STATS, insight)
    assert evidence.passed


def test_correlation_claim():
    insight = InsightItem(
        claim="Amount and distance from home are weakly correlated.",
        statistic="correlation", source_column="amount",
        comparison_column="distance_from_home", claimed_value=0.03,
    )
    evidence = evaluate_insight(PROFILE, STATS, insight)
    assert evidence.passed


def test_group_mean_claim():
    insight = InsightItem(
        claim="Non-fraud transactions average 65.94.",
        statistic="group_mean", source_column="fraud_flag",
        comparison_column="amount", category_value="0", claimed_value=65.939,
    )
    evidence = evaluate_insight(PROFILE, STATS, insight)
    assert evidence.passed


def test_unlocatable_claim_fails_as_unverifiable():
    insight = InsightItem(
        claim="Some claim about a column that has no matching statistic.",
        statistic="mean", source_column="nonexistent_column", claimed_value=1.0,
    )
    evidence = evaluate_insight(PROFILE, STATS, insight)
    assert not evidence.passed
    assert evidence.actual_value is None


def test_evaluate_all_handles_malformed_insight_dict():
    malformed = [{"claim": "bad insight", "statistic": "not_a_real_stat_type"}]
    evidence = evaluate_all(PROFILE, STATS, malformed)
    assert len(evidence) == 1
    assert not evidence[0].passed
