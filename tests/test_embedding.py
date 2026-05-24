"""Tests for embedding computation."""

import polars as pl
import pytest

from associo.embedding import embedding, spectral_embedding, node2vec


def _similarity_df():
    items = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i < j:
                rows.append({"lhs": a, "rhs": b, "similarity": 1.0 / abs(i - j)})
    return pl.DataFrame(rows)


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


def test_spectral_embedding():
    df = _similarity_df()
    result = spectral_embedding(
        df, column_lhs="lhs", column_rhs="rhs", column_similarity="similarity", dim=4
    )
    assert result.columns == ["item", "embedding"]
    assert result.shape[0] == 6
    assert all(len(v) == 4 for v in result["embedding"].to_list())


def test_spectral_embedding_empty():
    df = pl.DataFrame(schema={"lhs": pl.Utf8, "rhs": pl.Utf8, "similarity": pl.Float64})
    result = spectral_embedding(
        df, column_lhs="lhs", column_rhs="rhs", column_similarity="similarity"
    )
    assert result.shape[0] == 0
    assert result.columns == ["item", "embedding"]


def test_node2vec():
    df = _similarity_df()
    result = node2vec(
        df,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
        dim=8,
        num_walks=5,
        walk_length=10,
        n_iter=2,
    )
    assert result.columns == ["item", "embedding"]
    assert result.shape[0] == 6
    assert all(len(v) == 8 for v in result["embedding"].to_list())


def test_node2vec_deterministic():
    df = _similarity_df()
    kwargs = dict(
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
        dim=8,
        num_walks=5,
        walk_length=10,
        n_iter=2,
    )
    first = node2vec(df, **kwargs).sort("item")
    second = node2vec(df, **kwargs).sort("item")
    assert first["embedding"].to_list() == second["embedding"].to_list()
