"""Embeddings for items: t-SNE, spectral decomposition, and Node2Vec."""

from __future__ import annotations

import networkx as nx
import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.manifold import TSNE

from associo._matrix import pairwise_matrix
from associo._validation import validate_columns
from associo.graph import _build_graph

_EMBED_SCHEMA = {"item": pl.Utf8, "embedding": pl.List(pl.Float64)}


def _embedding_result(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(records, schema=_EMBED_SCHEMA)


def embedding(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_distance: str,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> pl.DataFrame:
    """Generate 2D coordinates for items using t-SNE on a precomputed distance matrix.

    Parameters
    ----------
    df : Edge list with (lhs, rhs, distance).
    column_lhs / column_rhs : Column names for edge endpoints.
    column_distance : Column name for distance values.
    perplexity : t-SNE perplexity (5–50). Default 30.
    n_iter : Number of iterations (1000–2000). Default 1000.

    Returns
    -------
    DataFrame with columns ``item``, ``x``, ``y``.
    """
    validate_columns(df, [column_lhs, column_rhs, column_distance], func_name="embedding")

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Distance is symmetric; missing pairs default to 1.0, self-distance to 0.
    items, mat = pairwise_matrix(
        df, column_lhs, column_rhs, column_distance,
        fill_value=1.0, agg="min", zero_diagonal=True,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        init="random",
        metric="precomputed",
    )
    coords = tsne.fit_transform(mat)

    return pl.DataFrame({
        "item": items,
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
    })


# ---------------------------------------------------------------------------
# Spectral embedding
# ---------------------------------------------------------------------------

def _prepare_edges(
    df: pl.DataFrame,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
) -> pl.DataFrame:
    """Embedding-specific edge cleanup: drop self-loops, null endpoints and
    non-positive weights (the latter would break alias sampling / Laplacians).
    The generic graph construction is left to ``graph._build_graph``."""
    return df.filter(
        (pl.col(column_lhs) != pl.col(column_rhs))
        & (pl.col(column_similarity) > 0)
    )


def _embed_component(G: nx.Graph, dim: int, rng: np.random.Generator) -> list[dict]:
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 1:
        return [{"item": str(nodes[0]), "embedding": [0.0] * dim}]

    k_request = min(dim + 1, n - 1)

    L = nx.normalized_laplacian_matrix(G, nodelist=nodes, weight="weight")
    L = csr_matrix(L, dtype=np.float64)

    # Explicit starting vector keeps ARPACK deterministic without touching
    # the global numpy RNG state.
    v0 = rng.random(n)

    try:
        eigenvalues, eigenvectors = eigsh(
            L, k=k_request, sigma=0, which="LM", tol=1e-6, maxiter=5000, v0=v0,
        )
    except Exception:
        try:
            eigenvalues, eigenvectors = eigsh(
                L, k=k_request, which="SM", tol=1e-6, maxiter=5000, v0=v0,
            )
        except Exception:
            L_dense = L.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[:k_request]
            eigenvectors = eigenvectors[:, :k_request]

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Drop the trivial (near-zero) eigenvalue components
    non_trivial = eigenvalues > 1e-8
    emb = eigenvectors[:, non_trivial][:, :dim]

    # Deterministic sign convention per axis
    for j in range(emb.shape[1]):
        idx = np.argmax(np.abs(emb[:, j]))
        if emb[idx, j] < 0:
            emb[:, j] = -emb[:, j]

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    emb = emb / norms

    if emb.shape[1] < dim:
        pad = np.zeros((emb.shape[0], dim - emb.shape[1]))
        emb = np.hstack([emb, pad])

    return [
        {"item": str(node), "embedding": emb[i].tolist()}
        for i, node in enumerate(nodes)
    ]


def spectral_embedding(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    dim: int = 16,
) -> pl.DataFrame:
    """Generate item embeddings via spectral decomposition of the graph.

    Uses the eigenvectors of the normalized graph Laplacian. Each connected
    component is embedded independently.

    Parameters
    ----------
    df : Edge list with (lhs, rhs, similarity).
    column_lhs / column_rhs : Column names for edge endpoints.
    column_similarity : Column name for similarity weight.
    dim : Embedding dimensionality. Default 16.

    Returns
    -------
    DataFrame with columns ``item``, ``embedding`` (List[Double]).
    """
    validate_columns(
        df, [column_lhs, column_rhs, column_similarity], func_name="spectral_embedding"
    )

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    rng = np.random.default_rng(42)
    edges = _prepare_edges(df, column_lhs, column_rhs, column_similarity)
    G = _build_graph(edges, column_lhs, column_rhs, column_similarity)

    if G.number_of_nodes() == 0:
        return _embedding_result([])

    if G.number_of_nodes() == 1:
        node = next(iter(G.nodes()))
        return _embedding_result([{"item": str(node), "embedding": [0.0] * dim}])

    records: list[dict] = []
    for component in nx.connected_components(G):
        records.extend(_embed_component(G.subgraph(component), dim, rng))

    return _embedding_result(records)


# ---------------------------------------------------------------------------
# Node2Vec (biased random walks + skip-gram with negative sampling)
# ---------------------------------------------------------------------------

def _alias_setup(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vose's alias method for O(1) sampling."""
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)
    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        (smaller if q[kk] < 1.0 else larger).append(kk)
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        (smaller if q[large] < 1.0 else larger).append(large)
    return J, q


def _alias_draw(J: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> int:
    K = len(J)
    kk = rng.integers(K)
    return kk if rng.random() < q[kk] else J[kk]


def _precompute_node_aliases(G: nx.Graph, nodes: list) -> dict:
    """Start-step distribution: probabilities proportional to edge weights."""
    alias_nodes = {}
    for node in nodes:
        neighbors = sorted(G.neighbors(node))
        if not neighbors:
            alias_nodes[node] = (None, None, [])
            continue
        weights = np.array([G[node][nbr].get("weight", 1.0) for nbr in neighbors])
        J, q = _alias_setup(weights / weights.sum())
        alias_nodes[node] = (J, q, neighbors)
    return alias_nodes


def _get_alias_edge(G: nx.Graph, src, dst, p: float, q: float) -> tuple:
    """Transition distribution dst -> next given we arrived from src."""
    neighbors = sorted(G.neighbors(dst))
    if not neighbors:
        return (None, None, [])
    unnormalized = []
    for nbr in neighbors:
        w = G[dst][nbr].get("weight", 1.0)
        if nbr == src:
            unnormalized.append(w / p)          # return
        elif G.has_edge(nbr, src):
            unnormalized.append(w)              # neighbour of source (BFS)
        else:
            unnormalized.append(w / q)          # away from source (DFS)
    unnormalized = np.array(unnormalized)
    J, qq = _alias_setup(unnormalized / unnormalized.sum())
    return (J, qq, neighbors)


def _precompute_edge_aliases(G: nx.Graph, p: float, q: float) -> dict:
    alias_edges = {}
    for src, dst in G.edges():
        alias_edges[(src, dst)] = _get_alias_edge(G, src, dst, p, q)
        alias_edges[(dst, src)] = _get_alias_edge(G, dst, src, p, q)
    return alias_edges


def _node2vec_walk(
    start, walk_length: int, alias_nodes: dict, alias_edges: dict,
    rng: np.random.Generator,
) -> list:
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        J, q, neighbors = alias_nodes[cur]
        if not neighbors:
            break
        if len(walk) == 1:
            walk.append(neighbors[_alias_draw(J, q, rng)])
        else:
            prev = walk[-2]
            J_e, q_e, nbrs_e = alias_edges[(prev, cur)]
            if not nbrs_e:
                break
            walk.append(nbrs_e[_alias_draw(J_e, q_e, rng)])
    return walk


def _generate_walks(
    G: nx.Graph, nodes: list, num_walks: int, walk_length: int,
    alias_nodes: dict, alias_edges: dict, rng: np.random.Generator,
) -> list:
    walks = []
    for _ in range(num_walks):
        shuffled = list(nodes)
        rng.shuffle(shuffled)
        for start in shuffled:
            walk = _node2vec_walk(start, walk_length, alias_nodes, alias_edges, rng)
            if len(walk) > 1:
                walks.append(walk)
    return walks


def _train_skipgram(
    walks: list, node_to_idx: dict, n: int, dim: int,
    window: int, n_iter: int, lr: float, rng: np.random.Generator,
) -> np.ndarray:
    W_in = (rng.random((n, dim)) - 0.5) / dim
    W_out = np.zeros((n, dim))

    # Negative sampling distribution ~ unigram^0.75
    counts = np.zeros(n)
    for walk in walks:
        for node in walk:
            counts[node_to_idx[node]] += 1
    neg_dist = np.power(counts + 1e-10, 0.75)
    neg_dist /= neg_dist.sum()
    neg_table_size = min(10_000_000, max(100_000, n * 100))
    neg_table = rng.choice(n, size=neg_table_size, p=neg_dist)
    neg_ptr = 0
    n_negative = 5

    for _ in range(n_iter):
        rng.shuffle(walks)
        for walk in walks:
            walk_idx = [node_to_idx[w] for w in walk]
            for i, center in enumerate(walk_idx):
                lo = max(0, i - window)
                hi = min(len(walk_idx), i + window + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    context = walk_idx[j]

                    v_in = W_in[center]
                    v_out_pos = W_out[context]
                    score = np.dot(v_in, v_out_pos)
                    sig = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
                    grad_pos = sig - 1.0

                    if neg_ptr + n_negative >= neg_table_size:
                        neg_ptr = 0
                    negs = neg_table[neg_ptr:neg_ptr + n_negative]
                    neg_ptr += n_negative

                    grad_in = grad_pos * v_out_pos.copy()
                    W_out[context] -= lr * grad_pos * v_in

                    for neg in negs:
                        if neg == center or neg == context:
                            continue
                        v_out_neg = W_out[neg]
                        score_n = np.dot(v_in, v_out_neg)
                        sig_n = 1.0 / (1.0 + np.exp(-np.clip(score_n, -30, 30)))
                        grad_neg = sig_n  # target = 0
                        grad_in += grad_neg * v_out_neg
                        W_out[neg] -= lr * grad_neg * v_in

                    W_in[center] -= lr * grad_in

    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return W_in / norms


def node2vec(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_similarity: str,
    dim: int = 64,
    walk_length: int = 30,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 5,
    n_iter: int = 5,
    learning_rate: float = 0.025,
    min_edge_weight: float = 0.0,
) -> pl.DataFrame:
    """Generate item embeddings using Node2Vec (biased random walks + skip-gram).

    Pure numpy + networkx implementation with no external ML dependencies.
    Alias sampling (Vose) gives O(1) transitions; skip-gram uses negative
    sampling. Output is L2-normalized for cosine similarity. With p=q=1 this
    reduces to DeepWalk.

    Parameters
    ----------
    df : Edge list with (lhs, rhs, similarity).
    column_lhs / column_rhs : Column names for edge endpoints.
    column_similarity : Column name for similarity weight.
    dim : Embedding dimensionality. Default 64.
    walk_length : Length of each random walk. Default 30.
    num_walks : Walks started from each node. Default 10.
    p : Return parameter. p<1 favours backtracking (BFS-like).
    q : In-out parameter. q<1 favours DFS, q>1 favours BFS.
    window : Skip-gram window size. Default 5.
    n_iter : Skip-gram training epochs. Default 5.
    learning_rate : Skip-gram learning rate. Default 0.025.
    min_edge_weight : Minimum edge weight to include. Default 0.0.

    Returns
    -------
    DataFrame with columns ``item``, ``embedding`` (List[Double]).
    """
    validate_columns(
        df, [column_lhs, column_rhs, column_similarity], func_name="node2vec"
    )

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    rng = np.random.default_rng(42)
    edges = _prepare_edges(df, column_lhs, column_rhs, column_similarity)
    G = _build_graph(edges, column_lhs, column_rhs, column_similarity, min_edge_weight)

    if G.number_of_nodes() == 0:
        return _embedding_result([])

    if G.number_of_nodes() == 1:
        node = next(iter(G.nodes()))
        return _embedding_result([{"item": str(node), "embedding": [0.0] * dim}])

    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    alias_nodes = _precompute_node_aliases(G, nodes)
    alias_edges = _precompute_edge_aliases(G, p, q)

    walks = _generate_walks(G, nodes, num_walks, walk_length, alias_nodes, alias_edges, rng)
    embeddings = _train_skipgram(
        walks, node_to_idx, len(nodes), dim, window, n_iter, learning_rate, rng
    )

    records = [
        {"item": str(node), "embedding": embeddings[node_to_idx[node]].tolist()}
        for node in nodes
    ]
    return _embedding_result(records)
