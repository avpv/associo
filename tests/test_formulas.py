"""Comprehensive formula-level tests against manually computed reference values.

Reference: lhs_rhs=30, lhs=50, rhs=40, total=100

Contingency table:
                RHS=1   RHS=0   Total
    LHS=1       30      20      50
    LHS=0       10      40      50
    Total       40      60      100

All expected values are computed by hand to match the original SQL formulas.
"""

import math

import polars as pl
import pytest

from associo.measures import association_measures


@pytest.fixture
def ref():
    """Reference counts and their expected measures."""
    df = pl.DataFrame({
        "lhs_rhs_count": [30],
        "lhs_total_count": [50],
        "rhs_total_count": [40],
        "total_count": [100],
    })
    result = association_measures(df)
    return result.row(0, named=True)


# -- Contingency table cells --

def test_contingency_lhs_not_rhs(ref):
    assert ref["lhs_not_rhs_count"] == pytest.approx(20.0, abs=1e-5)


def test_contingency_not_lhs_rhs(ref):
    assert ref["not_lhs_rhs_count"] == pytest.approx(10.0, abs=1e-5)


def test_contingency_not_lhs_not_rhs(ref):
    assert ref["not_lhs_not_rhs_count"] == pytest.approx(40.0, abs=1e-5)


# -- Basic measures --

def test_support(ref):
    # 30 / 100
    assert ref["support"] == pytest.approx(0.3, abs=1e-5)


def test_coverage(ref):
    # 50 / 100
    assert ref["coverage"] == pytest.approx(0.5, abs=1e-5)


def test_prevalence(ref):
    # 40 / 100
    assert ref["prevalence"] == pytest.approx(0.4, abs=1e-5)


def test_confidence(ref):
    # 30 / 50
    assert ref["confidence"] == pytest.approx(0.6, abs=1e-5)


def test_reverse_confidence(ref):
    # 30 / 40
    assert ref["reverse_confidence"] == pytest.approx(0.75, abs=1e-5)


def test_lift(ref):
    # 0.6 / 0.4
    assert ref["lift"] == pytest.approx(1.5, abs=1e-5)


def test_leverage(ref):
    # 0.3 - 0.5 * 0.4 = 0.1
    assert ref["leverage"] == pytest.approx(0.1, abs=1e-5)


# -- Laplace confidence --

def test_confidence_laplace(ref):
    # (30 + 2) / (50 + 4) = 32/54
    assert ref["confidence_laplace"] == pytest.approx(32 / 54, abs=1e-5)


# -- Confidence intervals --

def test_confidence_intervals(ref):
    conf = 0.6
    se = math.sqrt(conf * (1 - conf) / 50)
    assert ref["confidence_lower"] == pytest.approx(conf - 1.96 * se, abs=1e-4)
    assert ref["confidence_upper"] == pytest.approx(conf + 1.96 * se, abs=1e-4)


# -- Directional measures --

def test_zhangs_metric(ref):
    # conf = 0.6, conf_not_lhs_to_rhs = 10/50 = 0.2
    # zhang = (0.6 - 0.2) / max(0.6, 0.2) = 0.4 / 0.6
    assert ref["zhangs_metric"] == pytest.approx(0.4 / 0.6, abs=1e-5)


def test_improvement(ref):
    # 0.6 - 10/50 = 0.6 - 0.2 = 0.4
    assert ref["improvement"] == pytest.approx(0.4, abs=1e-5)


def test_added_value(ref):
    # 0.6 - 0.4 = 0.2
    assert ref["added_value"] == pytest.approx(0.2, abs=1e-5)


# -- Similarity measures --

def test_cosine(ref):
    # 0.3 / sqrt(0.5 * 0.4)
    assert ref["cosine"] == pytest.approx(0.3 / math.sqrt(0.2), abs=1e-5)


def test_jaccard(ref):
    # 0.3 / (0.5 + 0.4 - 0.3)
    assert ref["jaccard"] == pytest.approx(0.5, abs=1e-5)


def test_kulczynski(ref):
    # 0.5 * (0.3/0.5 + 0.3/0.4) = 0.5 * (0.6 + 0.75) = 0.675
    assert ref["kulczynski"] == pytest.approx(0.675, abs=1e-5)


def test_sokal_sneath(ref):
    # 2*30 / (50 + 40) = 60/90
    assert ref["sokal_sneath"] == pytest.approx(60 / 90, abs=1e-4)


def test_sokal_michener(ref):
    # (30 + 40) / 100 = 0.7
    assert ref["sokal_michener"] == pytest.approx(0.7, abs=1e-5)


def test_hamming(ref):
    # (50 + 40 - 2*30) / 100 = 30/100 = 0.3
    assert ref["hamming"] == pytest.approx(0.3, abs=1e-5)


# -- Statistical measures --

def test_conviction(ref):
    # (1 - 0.4) / (1 - 0.6) = 0.6 / 0.4 = 1.5
    assert ref["conviction"] == pytest.approx(1.5, abs=1e-5)


def test_odds_ratio(ref):
    # Haldane correction: (30.5 * 40.5) / (20.5 * 10.5)
    expected = (30.5 * 40.5) / (20.5 * 10.5)
    assert ref["odds_ratio"] == pytest.approx(expected, abs=1e-3)


def test_yules_q(ref):
    OR = (30.5 * 40.5) / (20.5 * 10.5)
    expected = (OR - 1) / (OR + 1)
    assert ref["yules_q"] == pytest.approx(expected, abs=1e-3)


def test_yules_y(ref):
    OR = (30.5 * 40.5) / (20.5 * 10.5)
    expected = (math.sqrt(OR) - 1) / (math.sqrt(OR) + 1)
    assert ref["yules_y"] == pytest.approx(expected, abs=1e-3)


def test_phi_coefficient(ref):
    # (S - C*P) / sqrt(C*(1-C)*P*(1-P))
    num = 0.3 - 0.5 * 0.4
    den = math.sqrt(0.5 * 0.5 * 0.4 * 0.6)
    assert ref["phi_coefficient"] == pytest.approx(num / den, abs=1e-4)


def test_chi_squared(ref):
    # Expected: 30*0.5*0.4=20, etc.
    cells = [
        (30, 50 * 0.4),   # ab, E(ab)=20
        (20, 50 * 0.6),   # a_not_b, E=30
        (10, 50 * 0.4),   # not_a_b, E=20
        (40, 50 * 0.6),   # not_a_not_b, E=30
    ]
    expected = sum((o - e) ** 2 / e for o, e in cells)
    assert ref["chi_squared"] == pytest.approx(expected, abs=1e-3)


def test_kappa(ref):
    obs = 0.3 + (1 - 0.5 - 0.4 + 0.3)  # 0.7
    exp = 0.5 * 0.4 + 0.5 * 0.6  # 0.5
    assert ref["kappa"] == pytest.approx((obs - exp) / (1 - exp), abs=1e-4)


def test_certainty_factor(ref):
    # (0.6 - 0.4) / (1 - 0.4) = 0.2/0.6
    assert ref["certainty_factor"] == pytest.approx(0.2 / 0.6, abs=1e-4)


def test_relative_risk(ref):
    # conf / conf_not_lhs_to_rhs = 0.6 / 0.2 = 3.0
    assert ref["relative_risk"] == pytest.approx(3.0, abs=1e-5)


def test_standardized_lift(ref):
    # (1.5 - 1) / (1.5 + 1) = 0.5/2.5 = 0.2
    assert ref["standardized_lift"] == pytest.approx(0.2, abs=1e-5)


def test_varying_rates_liaison(ref):
    # lift - 1 = 0.5
    assert ref["varying_rates_liaison"] == pytest.approx(0.5, abs=1e-5)


def test_rule_power_factor(ref):
    # 0.3 * 0.6 = 0.18
    assert ref["rule_power_factor"] == pytest.approx(0.18, abs=1e-5)


def test_confirmed_confidence(ref):
    # conf - conf_lhs_to_not_rhs = 0.6 - 20/50 = 0.6 - 0.4 = 0.2
    assert ref["confirmed_confidence"] == pytest.approx(0.2, abs=1e-5)


def test_difference_of_confidence(ref):
    # conf - conf_not_lhs_to_rhs = 0.6 - 0.2 = 0.4
    assert ref["difference_of_confidence"] == pytest.approx(0.4, abs=1e-5)


def test_casual_confidence(ref):
    # 0.5 * (conf + not_lhs_not_rhs/not_lhs_total)
    # = 0.5 * (0.6 + 40/50) = 0.5 * (0.6 + 0.8) = 0.7
    assert ref["casual_confidence"] == pytest.approx(0.7, abs=1e-5)


def test_imbalance_ratio(ref):
    # (coverage - prevalence) / (coverage + prevalence - support)
    # = (0.5 - 0.4) / (0.5 + 0.4 - 0.3) = 0.1/0.6
    assert ref["imbalance_ratio"] == pytest.approx(0.1 / 0.6, abs=1e-4)


def test_sebag_schoenauer(ref):
    # conf / conf_lhs_to_not_rhs = 0.6 / 0.4 = 1.5
    assert ref["sebag_schoenauer"] == pytest.approx(1.5, abs=1e-5)


def test_fisher_transformation_confidence(ref):
    # 2 * arcsin(sqrt(0.6))
    expected = 2 * math.asin(math.sqrt(0.6))
    assert ref["fisher_transformation_confidence"] == pytest.approx(expected, abs=1e-4)


def test_fisher_transformation_reverse_confidence(ref):
    # 2 * arcsin(sqrt(0.75))
    expected = 2 * math.asin(math.sqrt(0.75))
    assert ref["fisher_transformation_reverse_confidence"] == pytest.approx(expected, abs=1e-4)


# -- Validation tests --

def test_missing_column_raises():
    """Clear error when a required column is missing."""
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    with pytest.raises(ValueError, match="missing columns"):
        association_measures(df)


def test_missing_column_mentions_function_name():
    df = pl.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="association_measures"):
        association_measures(df)
