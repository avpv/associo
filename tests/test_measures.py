"""Tests for association measures calculation."""

import math

import polars as pl
import pytest

from associo.measures import calculate_association_measures, ALL_MEASURES


@pytest.fixture
def simple_counts():
    """Simple 2x2 contingency: lhs_rhs=30, lhs=50, rhs=40, total=100."""
    return pl.DataFrame({
        "lhs": ["A"],
        "rhs": ["B"],
        "lhs_rhs_count": [30],
        "lhs_total_count": [50],
        "rhs_total_count": [40],
        "total_count": [100],
    })


def test_basic_measures(simple_counts):
    result = calculate_association_measures(simple_counts)

    assert result.shape[0] == 1
    row = result.row(0, named=True)

    # support = 30/100 = 0.3
    assert row["support"] == pytest.approx(0.3, abs=1e-5)
    # coverage = 50/100 = 0.5
    assert row["coverage"] == pytest.approx(0.5, abs=1e-5)
    # prevalence = 40/100 = 0.4
    assert row["prevalence"] == pytest.approx(0.4, abs=1e-5)
    # confidence = 30/50 = 0.6
    assert row["confidence"] == pytest.approx(0.6, abs=1e-5)
    # reverse_confidence = 30/40 = 0.75
    assert row["reverse_confidence"] == pytest.approx(0.75, abs=1e-5)
    # lift = 0.6 / 0.4 = 1.5
    assert row["lift"] == pytest.approx(1.5, abs=1e-5)
    # leverage = 0.3 - 0.5*0.4 = 0.1
    assert row["leverage"] == pytest.approx(0.1, abs=1e-5)


def test_contingency_cells(simple_counts):
    result = calculate_association_measures(simple_counts)
    row = result.row(0, named=True)

    # lhs_not_rhs = 50 - 30 = 20
    assert row["lhs_not_rhs_count"] == pytest.approx(20, abs=1e-5)
    # not_lhs_rhs = 40 - 30 = 10
    assert row["not_lhs_rhs_count"] == pytest.approx(10, abs=1e-5)
    # not_lhs_not_rhs = 100 - 50 - 40 + 30 = 40
    assert row["not_lhs_not_rhs_count"] == pytest.approx(40, abs=1e-5)


def test_jaccard(simple_counts):
    result = calculate_association_measures(simple_counts)
    row = result.row(0, named=True)
    # jaccard = 0.3 / (0.5 + 0.4 - 0.3) = 0.3/0.6 = 0.5
    assert row["jaccard"] == pytest.approx(0.5, abs=1e-5)


def test_cosine(simple_counts):
    result = calculate_association_measures(simple_counts)
    row = result.row(0, named=True)
    expected = 0.3 / math.sqrt(0.5 * 0.4)
    assert row["cosine"] == pytest.approx(expected, abs=1e-5)


def test_multiple_rows():
    df = pl.DataFrame({
        "lhs": ["A", "C"],
        "rhs": ["B", "D"],
        "lhs_rhs_count": [30, 10],
        "lhs_total_count": [50, 20],
        "rhs_total_count": [40, 80],
        "total_count": [100, 100],
    })
    result = calculate_association_measures(df)
    assert result.shape[0] == 2

    row0 = result.row(0, named=True)
    row1 = result.row(1, named=True)
    assert row0["support"] == pytest.approx(0.3, abs=1e-5)
    assert row1["support"] == pytest.approx(0.1, abs=1e-5)


def test_zero_denominator():
    """When lhs_total_count is 0, confidence should be null."""
    df = pl.DataFrame({
        "lhs": ["X"],
        "rhs": ["Y"],
        "lhs_rhs_count": [0],
        "lhs_total_count": [0],
        "rhs_total_count": [10],
        "total_count": [100],
    })
    result = calculate_association_measures(df)
    row = result.row(0, named=True)
    assert row["confidence"] is None
    assert row["support"] == pytest.approx(0.0, abs=1e-5)


def test_all_columns_present(simple_counts):
    result = calculate_association_measures(simple_counts)
    expected_cols = {
        "support", "coverage", "prevalence", "confidence", "reverse_confidence",
        "lift", "leverage", "confidence_laplace", "confidence_lower", "confidence_upper",
        "zhangs_metric", "importance", "added_value", "improvement", "cosine",
        "conviction", "jaccard", "kulczynski", "klosgen", "relative_difference",
        "rule_power_factor", "weight_of_evidence", "casual_confidence", "casual_support",
        "confirmed_confidence", "odds_ratio", "certainty_factor", "chi_squared",
        "phi_coefficient", "yules_q", "yules_y", "kappa", "j_measure",
        "hamming", "sokal_michener", "sokal_sneath",
        "fisher_transformation_confidence", "fisher_transformation_reverse_confidence",
    }
    assert expected_cols.issubset(set(result.columns))


def test_selective_measures(simple_counts):
    """Only compute a subset of measures."""
    result = calculate_association_measures(
        simple_counts, measures=["support", "confidence", "lift"],
    )
    row = result.row(0, named=True)
    assert "support" in result.columns
    assert "confidence" in result.columns
    assert "lift" in result.columns
    # Should NOT have other measures
    assert "jaccard" not in result.columns
    assert "odds_ratio" not in result.columns


def test_unknown_measure_raises(simple_counts):
    with pytest.raises(ValueError, match="Unknown measures"):
        calculate_association_measures(simple_counts, measures=["nonexistent_metric"])


def test_all_measures_constant():
    assert "support" in ALL_MEASURES
    assert "lift" in ALL_MEASURES
    assert len(ALL_MEASURES) > 60
