"""Graph-based algorithms: clustering, communities, cliques, components, label propagation."""

from __future__ import annotations

import random
from collections import Counter

import networkx as nx
import polars as pl
from sklearn.cluster import AffinityPropagation
import numpy as np

from associo._validation import validate_columns


def _build_graph(
    df: pl.DataFrame | pl.LazyFrame,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    min_edge_weight: float = 0.0,
    *,
    accumulate_weights: bool = False,
    drop_self_loops: bool = False,
    require_positive: bool = False,
) -> nx.Graph:
    """Build a NetworkX graph from a Polars DataFrame of edges.

    Parameters
    ----------
    min_edge_weight : Keep only edges with ``similarity >= min_edge_weight``.
    accumulate_weights : Sum weights of duplicate (lhs, rhs) pairs instead of
        keeping the last one. Needed by embedding algorithms.
    drop_self_loops : Drop edges where lhs == rhs (and null endpoints).
    require_positive : Additionally drop edges with ``similarity <= 0``.
    """
    validate_columns(df, [column_lhs, column_rhs, column_similarity], func_name="_build_graph")

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    predicate = pl.col(column_similarity).is_not_null() & (
        pl.col(column_similarity) >= min_edge_weight
    )
    if require_positive:
        predicate = predicate & (pl.col(column_similarity) > 0)
    if drop_self_loops:
        predicate = (
            predicate
            & pl.col(column_lhs).is_not_null()
            & pl.col(column_rhs).is_not_null()
            & (pl.col(column_lhs) != pl.col(column_rhs))
        )

    # Filter and extract columns directly — avoids slow iter_rows
    filtered = df.filter(predicate)
    lhs_col = filtered[column_lhs].to_list()
    rhs_col = filtered[column_rhs].to_list()
    w_col = filtered[column_similarity].to_list()

    G = nx.Graph()
    if accumulate_weights:
        for lhs, rhs, w in zip(lhs_col, rhs_col, w_col):
            if G.has_edge(lhs, rhs):
                G[lhs][rhs]["weight"] += w
            else:
                G.add_edge(lhs, rhs, weight=w)
    else:
        G.add_weighted_edges_from(zip(lhs_col, rhs_col, w_col))
    return G


def _result_df(records: list[dict], columns: list[str]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema={c: pl.Utf8 for c in columns})
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# Affinity Propagation clustering
# ---------------------------------------------------------------------------

def clusters(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    cluster_preference_factor: int = 50,
) -> pl.DataFrame:
    """Cluster items using Affinity Propagation on a precomputed similarity matrix.

    Parameters
    ----------
    df : Edge list with (lhs, rhs, similarity).
    column_lhs / column_rhs : Column names for edge endpoints.
    column_similarity : Column name for similarity weight.
    cluster_preference_factor : Percentile of similarity values used as preference (0-100).
        Lower → fewer, larger clusters. Higher → more clusters.

    Returns
    -------
    DataFrame with columns ``item``, ``cluster_label``.
    """
    validate_columns(df, [column_lhs, column_rhs, column_similarity], func_name="clusters")

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Symmetrise edges: add reverse direction, then take max for each pair
    forward = df.select(
        pl.col(column_lhs).alias("a"),
        pl.col(column_rhs).alias("b"),
        pl.col(column_similarity).alias("sim"),
    )
    reverse = df.select(
        pl.col(column_rhs).alias("a"),
        pl.col(column_lhs).alias("b"),
        pl.col(column_similarity).alias("sim"),
    )
    sym = pl.concat([forward, reverse]).group_by("a", "b").agg(pl.col("sim").max()).sort("a", "b")

    # Collect all unique items
    all_items = sorted(
        set(sym["a"].to_list()) | set(sym["b"].to_list())
    )

    pivot = sym.pivot(on="b", index="a", values="sim").fill_null(0)
    # Ensure all items appear as both rows and columns
    for item in all_items:
        if item not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0.0).alias(item))
    pivot = pivot.sort("a")
    items = pivot["a"].to_list()
    mat = pivot.select(items).to_numpy()

    similarities = mat[mat > 0]
    pref = float(np.percentile(similarities, np.clip(cluster_preference_factor, 0, 100))) if len(similarities) > 0 else 0.0

    ap = AffinityPropagation(affinity="precomputed", preference=pref, random_state=42)
    labels = ap.fit_predict(mat)

    return pl.DataFrame({
        "item": items,
        "cluster_label": [str(l) for l in labels],
    })


# ---------------------------------------------------------------------------
# Louvain communities
# ---------------------------------------------------------------------------

def communities(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    community_resolution: float = 1.0,
) -> pl.DataFrame:
    """Detect communities using the Louvain algorithm.

    Parameters
    ----------
    community_resolution : Resolution parameter. Higher → more communities. Recommended 0.5–2.0.

    Returns
    -------
    DataFrame with columns ``item``, ``community_label``.
    """
    G = _build_graph(df, column_lhs, column_rhs, column_similarity)
    if len(G) == 0:
        return _result_df([], ["item", "community_label"])

    communities = nx.algorithms.community.louvain_communities(
        G, weight="weight", resolution=community_resolution, threshold=1e-7, seed=42,
    )

    records = []
    for idx, comm in enumerate(communities):
        for node in comm:
            records.append({"item": str(node), "community_label": str(idx)})

    return _result_df(records, ["item", "community_label"])


# ---------------------------------------------------------------------------
# Maximal cliques
# ---------------------------------------------------------------------------

def maximal_cliques(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    min_clique_size: int = 3,
    min_edge_weight: float = 0.1,
) -> pl.DataFrame:
    """Find all maximal cliques in the similarity graph.

    An item can belong to multiple cliques.

    Returns
    -------
    DataFrame with columns ``item``, ``clique_label``.
    """
    G = _build_graph(df, column_lhs, column_rhs, column_similarity, min_edge_weight)
    if len(G) == 0:
        return _result_df([], ["item", "clique_label"])

    records = []
    for idx, clique in enumerate(nx.find_cliques(G)):
        if len(clique) >= min_clique_size:
            for node in clique:
                records.append({"item": str(node), "clique_label": str(idx)})

    return _result_df(records, ["item", "clique_label"])


# ---------------------------------------------------------------------------
# K-clique communities
# ---------------------------------------------------------------------------

def k_clique_communities(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    k: int = 3,
    min_edge_weight: float = 0.1,
) -> pl.DataFrame:
    """Find overlapping communities using the k-clique percolation method.

    An item can belong to multiple communities.

    Returns
    -------
    DataFrame with columns ``item``, ``community_label``.
    """
    G = _build_graph(df, column_lhs, column_rhs, column_similarity, min_edge_weight)
    if len(G) == 0:
        return _result_df([], ["item", "community_label"])

    records = []
    try:
        communities = list(nx.algorithms.community.k_clique_communities(G, k=k))
        for idx, comm in enumerate(communities):
            for node in comm:
                records.append({"item": str(node), "community_label": str(idx)})
    except nx.NetworkXError:
        pass

    return _result_df(records, ["item", "community_label"])


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def connected_components(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    min_edge_weight: float = 0.0,
) -> pl.DataFrame:
    """Find connected components in the graph.

    Each item belongs to exactly one component.

    Returns
    -------
    DataFrame with columns ``item``, ``component_label``.
    """
    G = _build_graph(df, column_lhs, column_rhs, column_similarity, min_edge_weight)
    if len(G) == 0:
        return _result_df([], ["item", "component_label"])

    records = []
    for idx, component in enumerate(nx.connected_components(G)):
        for node in component:
            records.append({"item": str(node), "component_label": str(idx)})

    return _result_df(records, ["item", "component_label"])


# ---------------------------------------------------------------------------
# Label propagation (non-overlapping)
# ---------------------------------------------------------------------------

def label_propagation(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    min_edge_weight: float = 0.0,
) -> pl.DataFrame:
    """Detect communities using label propagation (non-overlapping).

    Fast algorithm suitable for large graphs. Each item belongs to exactly one community.

    Returns
    -------
    DataFrame with columns ``item``, ``community_label``.
    """
    G = _build_graph(df, column_lhs, column_rhs, column_similarity, min_edge_weight)
    if len(G) == 0:
        return _result_df([], ["item", "community_label"])

    communities = nx.algorithms.community.label_propagation_communities(G)

    records = []
    for idx, comm in enumerate(communities):
        for node in comm:
            records.append({"item": str(node), "community_label": str(idx)})

    return _result_df(records, ["item", "community_label"])


# ---------------------------------------------------------------------------
# Label propagation overlapping (SLPA-like)
# ---------------------------------------------------------------------------

def label_propagation_overlapping(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    min_edge_weight: float = 0.0,
    n_iter: int = 20,
    threshold: float = 0.1,
) -> pl.DataFrame:
    """Detect overlapping communities using an SLPA-like algorithm.

    An item can belong to multiple communities.

    Parameters
    ----------
    n_iter : Number of propagation iterations (default 20).
    threshold : Minimum label frequency ratio to keep (0.0–1.0). Lower → more overlap.

    Returns
    -------
    DataFrame with columns ``item``, ``community_label``.
    """
    G = _build_graph(df, column_lhs, column_rhs, column_similarity, min_edge_weight)
    if len(G) == 0:
        return _result_df([], ["item", "community_label"])

    rng = random.Random(42)
    nodes = list(G.nodes())
    memory: dict[str, list] = {node: [node] for node in nodes}

    for _ in range(n_iter):
        rng.shuffle(nodes)
        for node in nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            neighbor_labels: list = []
            for nb in neighbors:
                w = G[node][nb].get("weight", 1.0)
                if memory[nb]:
                    label = rng.choice(memory[nb])
                    neighbor_labels.extend([label] * int(w * 10 + 1))

            if neighbor_labels:
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                memory[node].append(most_common)

    # Post-processing: assign community labels
    label_to_id: dict = {}
    counter = 0
    records = []

    for node in G.nodes():
        counts = Counter(memory[node])
        total = len(memory[node])
        kept = [lab for lab, cnt in counts.items() if cnt / total >= threshold]
        for lab in kept:
            if lab not in label_to_id:
                label_to_id[lab] = str(counter)
                counter += 1
            records.append({"item": str(node), "community_label": label_to_id[lab]})

    return _result_df(records, ["item", "community_label"])
