"""Associo — high-performance association analysis library built on Polars."""

from associo.measures import association_measures, ALL_MEASURES
from associo.associations import direct_associations, combinatorial_associations
from associo.graph import (
    clusters,
    communities,
    maximal_cliques,
    k_clique_communities,
    connected_components,
    label_propagation,
    label_propagation_overlapping,
)
from associo.embedding import embedding, spectral_embedding, node2vec

__all__ = [
    "ALL_MEASURES",
    "association_measures",
    "direct_associations",
    "combinatorial_associations",
    "clusters",
    "communities",
    "maximal_cliques",
    "k_clique_communities",
    "connected_components",
    "label_propagation",
    "label_propagation_overlapping",
    "embedding",
    "spectral_embedding",
    "node2vec",
]
