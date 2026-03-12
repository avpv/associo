"""Tests for direct and combinatorial association computation."""

import polars as pl
import pytest

from associo.associations import compute_direct_associations, compute_combinatorial_associations


@pytest.fixture
def transaction_data():
    """Market basket-style data: 3 transactions with items."""
    return pl.DataFrame({
        "product": ["bread", "milk", "bread", "butter", "milk", "butter"],
        "category": ["food", "dairy", "food", "dairy", "dairy", "dairy"],
        "order_id": [1, 1, 2, 2, 3, 3],
    })


def test_direct_associations(transaction_data):
    result = compute_direct_associations(
        transaction_data,
        column_lhs="product",
        column_rhs="category",
        column_tid="order_id",
    )
    assert result.shape[0] > 0
    assert "support" in result.columns
    assert "confidence" in result.columns
    assert "lift" in result.columns

    # bread → food: appears in orders 1,2. bread in 2 orders, food in 2 orders, total 3
    bread_food = result.filter(
        (pl.col("lhs") == "bread") & (pl.col("rhs") == "food")
    )
    assert bread_food.shape[0] == 1
    row = bread_food.row(0, named=True)
    assert row["confidence"] == pytest.approx(1.0, abs=1e-5)  # bread always in food


def test_direct_associations_all_pairs(transaction_data):
    result = compute_direct_associations(
        transaction_data,
        column_lhs="product",
        column_rhs="category",
        column_tid="order_id",
        all_pairs=True,
    )
    # 3 products × 2 categories = 6 pairs (including non-co-occurring)
    assert result.shape[0] == 6


def test_combinatorial_associations():
    data = pl.DataFrame({
        "item": ["A", "B", "A", "C", "B", "C"],
        "tid": [1, 1, 2, 2, 3, 3],
    })
    result = compute_combinatorial_associations(
        data, column_items="item", column_tid="tid",
    )
    assert result.shape[0] > 0
    # Self-pairs excluded by default
    self_pairs = result.filter(pl.col("lhs") == pl.col("rhs"))
    assert self_pairs.shape[0] == 0

    # A,B,C each appear in 2 out of 3 transactions
    # All pairs co-occur in 1 out of 3 transactions
    ab = result.filter((pl.col("lhs") == "A") & (pl.col("rhs") == "B"))
    assert ab.shape[0] == 1
    assert ab.row(0, named=True)["support"] == pytest.approx(1 / 3, abs=1e-5)


def test_combinatorial_with_self_pairs():
    data = pl.DataFrame({
        "item": ["X", "Y"],
        "tid": [1, 1],
    })
    result = compute_combinatorial_associations(
        data, column_items="item", column_tid="tid", include_self_pairs=True,
    )
    # Should include X→X, X→Y, Y→X, Y→Y
    assert result.shape[0] == 4


def test_combinatorial_without_self_pairs():
    data = pl.DataFrame({
        "item": ["X", "Y"],
        "tid": [1, 1],
    })
    result = compute_combinatorial_associations(
        data, column_items="item", column_tid="tid",
    )
    # Only X→Y, Y→X
    assert result.shape[0] == 2
