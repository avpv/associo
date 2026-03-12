"""Tests for embedding computation."""

import polars as pl
import pytest

from associo.embedding import embedding


def test_embedding():
    # 5 items needed (perplexity < n_samples)
    items = ["A", "B", "C", "D", "E"]
    rows = []
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            dist = 0.0 if i == j else abs(i - j) * 0.2
            rows.append({"lhs": a, "rhs": b, "distance": dist})

    df = pl.DataFrame(rows)
    result = embedding(
        df,
        column_lhs="lhs",
        column_rhs="rhs",
        column_distance="distance",
        perplexity=2.0,
        n_iter=300,
    )
    assert "item" in result.columns
    assert "x" in result.columns
    assert "y" in result.columns
    assert result.shape[0] == 5
