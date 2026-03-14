# Associo

High-performance association analysis library built on [Polars](https://pola.rs/). Computes **60+ association metrics**, performs **graph clustering** and **community detection**, and provides **t-SNE embedding** for visualization.

## Features

| Function | Description |
|---|---|
| `direct_associations` | Association metrics from pre-paired lhs/rhs data |
| `combinatorial_associations` | All item-pair combinations within transactions |
| `association_measures` | 60+ metrics from pre-aggregated counts |
| `clusters` | Affinity Propagation clustering |
| `communities` | Louvain community detection |
| `label_propagation` | Fast label propagation |
| `connected_components` | Connected components |
| `maximal_cliques` | All maximal cliques |
| `k_clique_communities` | K-clique percolation |
| `label_propagation_overlapping` | SLPA-based overlapping communities |
| `embedding` | t-SNE 2D projection from distance matrix |

## Installation

```bash
pip install git+https://github.com/avpv/associo.git
```

**Requirements:** Python >= 3.10

**Dependencies:** `polars>=0.20.0`, `networkx>=3.0`, `scikit-learn>=1.3.0`

## Quick Start

### 1. Association Rules — Direct

When your data already has `lhs`, `rhs`, and transaction id columns:

```python
import polars as pl
from associo import direct_associations

df = pl.DataFrame({
    "product_a": ["bread", "bread", "milk", "eggs"],
    "product_b": ["butter", "milk", "butter", "bread"],
    "order_id":  [1, 2, 3, 4],
})

result = direct_associations(
    df,
    column_lhs="product_a",
    column_rhs="product_b",
    column_tid="order_id",
)
```

### 2. Association Rules — Combinatorial

When each row contains one item per transaction — all pairs are generated automatically:

```python
from associo import combinatorial_associations

basket = pl.DataFrame({
    "product":  ["bread", "butter", "milk", "bread", "milk", "eggs"],
    "order_id": [1, 1, 1, 2, 2, 2],
})

result = combinatorial_associations(
    basket,
    column_items="product",
    column_tid="order_id",
)
```

### 3. Compute metrics from pre-aggregated counts

```python
from associo import association_measures

counts = pl.DataFrame({
    "lhs": ["bread"],
    "rhs": ["butter"],
    "lhs_rhs_count": [150],
    "lhs_total_count": [500],
    "rhs_total_count": [300],
    "total_count": [1000],
})

result = association_measures(counts)

# Compute only specific metrics
result = association_measures(counts, measures=["support", "confidence", "lift"])
```

### 4. Clustering & Community Detection

```python
from associo import clusters, communities, label_propagation

similarity = pl.DataFrame({
    "item_a": ["A", "A", "B", "C"],
    "item_b": ["B", "C", "C", "D"],
    "sim":    [0.8, 0.3, 0.7, 0.9],
})

# Affinity Propagation
cl = clusters(similarity, column_lhs="item_a", column_rhs="item_b", column_similarity="sim")

# Louvain communities
comm = communities(similarity, column_lhs="item_a", column_rhs="item_b", column_similarity="sim")

# Fast label propagation
lp = label_propagation(similarity, column_lhs="item_a", column_rhs="item_b", column_similarity="sim")
```

### 5. Embedding

```python
from associo import embedding

distances = pl.DataFrame({
    "item_a": ["A", "A", "B"],
    "item_b": ["B", "C", "C"],
    "dist":   [0.2, 0.7, 0.3],
})

coords = embedding(distances, column_lhs="item_a", column_rhs="item_b", column_distance="dist")
# Returns DataFrame with columns: item, x, y
```

## Association Metrics Reference

The library computes **60+ association measures** from a 2×2 contingency table:

### Core Metrics
- **Support** — P(X ∧ Y)
- **Confidence** — P(Y|X)
- **Lift** — P(Y|X) / P(Y)
- **Leverage** — P(X ∧ Y) − P(X)·P(Y)
- **Conviction** — (1 − P(Y)) / (1 − confidence)

### Similarity & Distance
- Jaccard, Cosine, Kulczynski, Sokal-Sneath, Sokal-Michener, Rogers-Tanimoto, Hamming

### Statistical Tests
- Chi-squared, Phi coefficient, Odds ratio (with CI), Yule's Q & Y, Kappa, p-value approximation

### Information-Theoretic
- Mutual information, J-measure, Weight of evidence

### Interestingness Measures
- Zhang's metric, Added value, Certainty factor, Conviction, Importance (log-ratio), Sebag-Schoenauer, Relative risk, and many more

### Smoothed Variants
- Laplace-smoothed confidence with confidence intervals

## Algorithm Parameters

| Algorithm | Parameter | Default | Effect |
|---|---|---|---|
| Affinity Propagation | `cluster_preference_factor` | 50 | 0–100; lower = fewer larger clusters |
| Louvain | `community_resolution` | 1.0 | Higher = more communities (0.5–2.0) |
| Maximal Cliques | `min_clique_size` | 3 | Minimum clique size to return |
| Maximal Cliques | `min_edge_weight` | 0.1 | Edge weight threshold |
| K-Clique | `k` | 3 | Clique size for percolation |
| K-Clique | `min_edge_weight` | 0.1 | Edge weight threshold |
| Connected Components | `min_edge_weight` | 0.0 | Edge weight threshold |
| Label Propagation | `min_edge_weight` | 0.0 | Edge weight threshold |
| SLPA Overlapping | `n_iter` | 20 | Propagation iterations |
| SLPA Overlapping | `threshold` | 0.1 | Label inclusion threshold (0–1) |
| t-SNE | `perplexity` | 30 | 5–50; local vs global structure |
| t-SNE | `n_iter` | 1000 | Iterations (1000–2000) |

## Project Structure

```
├── src/associo/
│   ├── __init__.py          # Public API
│   ├── measures.py          # 60+ association metrics (Polars expressions)
│   ├── associations.py      # Direct & combinatorial association extraction
│   ├── graph.py             # Clustering, communities, cliques, components
│   ├── embedding.py         # t-SNE dimensionality reduction
│   └── _validation.py       # Input validation
├── tests/
│   ├── test_measures.py
│   ├── test_associations.py
│   ├── test_graph.py
│   ├── test_embedding.py
│   ├── test_formulas.py
│   └── test_validation.py
└── pyproject.toml
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
