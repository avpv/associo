"""Tests for input validation across all public functions."""

import polars as pl
import pytest

from associo.associations import compute_direct_associations, compute_combinatorial_associations
from associo.graph import (
    compute_communities,
    compute_clusters,
    compute_connected_components,
)
from associo.embedding import compute_embedding


def test_direct_associations_missing_column():
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    with pytest.raises(ValueError, match="compute_direct_associations.*missing columns"):
        compute_direct_associations(df, column_lhs="x", column_rhs="b", column_tid="c")


def test_combinatorial_missing_column():
    df = pl.DataFrame({"item": ["A"], "tid": [1]})
    with pytest.raises(ValueError, match="compute_combinatorial_associations.*missing columns"):
        compute_combinatorial_associations(df, column_items="nonexistent", column_tid="tid")


def test_graph_missing_column():
    df = pl.DataFrame({"a": ["X"], "b": ["Y"], "w": [0.5]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_communities(df, column_lhs="a", column_rhs="b", column_similarity="nonexistent")


def test_clusters_missing_column():
    df = pl.DataFrame({"a": ["X"], "b": ["Y"], "w": [0.5]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_clusters(df, column_lhs="missing", column_rhs="b", column_similarity="w")


def test_embedding_missing_column():
    df = pl.DataFrame({"lhs": ["A"], "rhs": ["B"], "dist": [0.5]})
    with pytest.raises(ValueError, match="compute_embedding.*missing columns"):
        compute_embedding(df, column_lhs="lhs", column_rhs="rhs", column_distance="nonexistent")


def test_connected_components_missing_column():
    df = pl.DataFrame({"x": ["A"], "y": ["B"], "z": [0.1]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_connected_components(df, column_lhs="x", column_rhs="y", column_similarity="missing")


def test_validation_works_with_lazyframe():
    """Validation should work on LazyFrames too."""
    lf = pl.DataFrame({"a": [1]}).lazy()
    with pytest.raises(ValueError, match="missing columns"):
        compute_direct_associations(lf, column_lhs="x", column_rhs="y", column_tid="z")
