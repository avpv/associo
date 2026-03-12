"""Tests for graph-based algorithms."""

import polars as pl
import pytest

from associo.graph import (
    clusters,
    communities,
    maximal_cliques,
    k_clique_communities,
    connected_components,
    label_propagation,
    label_propagation_overlapping,
)


@pytest.fixture
def edge_data():
    """A small graph with two clear groups: {A,B,C} and {D,E,F}."""
    return pl.DataFrame({
        "lhs": ["A", "A", "B", "D", "D", "E", "A"],
        "rhs": ["B", "C", "C", "E", "F", "F", "D"],
        "similarity": [0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.1],
    })


def test_clusters(edge_data):
    result = clusters(
        edge_data,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
    )
    assert "item" in result.columns
    assert "cluster_label" in result.columns
    assert result.shape[0] > 0


def test_communities(edge_data):
    result = communities(
        edge_data,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
    )
    assert "item" in result.columns
    assert "community_label" in result.columns
    items = set(result["item"].to_list())
    assert {"A", "B", "C", "D", "E", "F"}.issubset(items)


def test_maximal_cliques(edge_data):
    result = maximal_cliques(
        edge_data,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
        min_clique_size=3,
        min_edge_weight=0.5,
    )
    assert "item" in result.columns
    assert "clique_label" in result.columns
    # {A,B,C} and {D,E,F} form cliques of size 3
    assert result.shape[0] >= 3


def test_k_clique_communities(edge_data):
    result = k_clique_communities(
        edge_data,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
        k=3,
        min_edge_weight=0.5,
    )
    assert "item" in result.columns
    assert "community_label" in result.columns


def test_connected_components(edge_data):
    result = connected_components(
        edge_data,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
        min_edge_weight=0.5,
    )
    assert "item" in result.columns
    assert "component_label" in result.columns
    # With min_weight=0.5, A-D edge (0.1) is excluded → 2 components
    labels = result.group_by("component_label").len()
    assert labels.shape[0] == 2


def test_label_propagation(edge_data):
    result = label_propagation(
        edge_data,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
    )
    assert "item" in result.columns
    assert "community_label" in result.columns
    assert result.shape[0] == 6


def test_label_propagation_overlapping(edge_data):
    result = label_propagation_overlapping(
        edge_data,
        column_lhs="lhs",
        column_rhs="rhs",
        column_similarity="similarity",
        n_iter=20,
        threshold=0.1,
    )
    assert "item" in result.columns
    assert "community_label" in result.columns
    assert result.shape[0] >= 6


def test_empty_graph():
    empty = pl.DataFrame({"lhs": [], "rhs": [], "similarity": []}).cast({
        "lhs": pl.Utf8, "rhs": pl.Utf8, "similarity": pl.Float64,
    })
    r1 = communities(empty, column_lhs="lhs", column_rhs="rhs", column_similarity="similarity")
    assert r1.shape[0] == 0
    r2 = connected_components(empty, column_lhs="lhs", column_rhs="rhs", column_similarity="similarity")
    assert r2.shape[0] == 0
