"""Associo — high-performance association analysis library built on Polars."""

from associo.measures import calculate_association_measures
from associo.associations import compute_direct_associations, compute_combinatorial_associations
from associo.graph import (
    compute_clusters,
    compute_communities,
    compute_maximal_cliques,
    compute_k_clique_communities,
    compute_connected_components,
    compute_label_propagation,
    compute_label_propagation_overlapping,
)
from associo.embedding import compute_embedding

__all__ = [
    "calculate_association_measures",
    "compute_direct_associations",
    "compute_combinatorial_associations",
    "compute_clusters",
    "compute_communities",
    "compute_maximal_cliques",
    "compute_k_clique_communities",
    "compute_connected_components",
    "compute_label_propagation",
    "compute_label_propagation_overlapping",
    "compute_embedding",
]
