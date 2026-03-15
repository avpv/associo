<p align="center">
  <img src="logo.svg" alt="Associo" width="120"/>
</p>

# Associo

*From Italian "associo" — I associate. A library for discovering associations and connections in data.*

High-performance association analysis library built on [Polars](https://pola.rs/). Computes **60+ association metrics**, performs **graph clustering** and **community detection**, and provides **t-SNE embedding** for visualization.

```bash
pip install git+https://github.com/avpv/associo.git
```

**Requirements:** Python >= 3.10 &nbsp;|&nbsp; **Dependencies:** `polars>=0.20.0`, `networkx>=3.0`, `scikit-learn>=1.3.0`

---

## How It Works

<p align="center">
  <img src="docs/img/pipeline.svg" alt="Associo Pipeline" width="780"/>
</p>

**All inputs and outputs are Polars DataFrames** — you can compose steps freely in any order.

---

## Table of Contents

1. [The Contingency Table — Foundation of All Metrics](#1-the-contingency-table)
2. [Core Metrics Explained Visually](#2-core-metrics-explained-visually)
   - [Support](#support--px--y)
   - [Confidence](#confidence--pyx)
   - [Lift](#lift--pyx--py)
   - [Jaccard Similarity](#jaccard-similarity)
   - [Leverage & Conviction](#leverage--conviction)
3. [All 60+ Metrics — Taxonomy](#3-all-60-metrics--taxonomy)
4. [Quick Start](#4-quick-start)
5. [API Reference](#5-api-reference)
6. [Graph Algorithms](#6-graph-algorithms)
7. [Embedding & Visualization](#7-embedding--visualization)
8. [Algorithm Parameters](#8-algorithm-parameters)

---

## 1. The Contingency Table

Every metric in Associo is derived from a **2×2 contingency table**. This table counts how often two items (X and Y) appear together and apart across all transactions.

<p align="center">
  <img src="docs/img/contingency_table.svg" alt="2×2 Contingency Table" width="520"/>
</p>

**How to read it:**

| Cell | Meaning | Example |
|------|---------|---------|
| **a,b = 30** | Both X and Y present | 30 orders have both bread and butter |
| **a,¬b = 20** | X present, Y absent | 20 orders have bread but not butter |
| **¬a,b = 10** | X absent, Y present | 10 orders have butter but not bread |
| **¬a,¬b = 40** | Neither present | 40 orders have neither |
| **a = 50** | Total transactions with X | bread appears in 50 orders |
| **b = 40** | Total transactions with Y | butter appears in 40 orders |
| **n = 100** | Grand total | 100 orders total |

> **Throughout this guide**, we use these example values: a,b=30, a=50, b=40, n=100.

---

## 2. Core Metrics Explained Visually

### Support = P(X ∧ Y)

**"How often do X and Y appear together?"**

<p align="center">
  <img src="docs/img/venn_support.svg" alt="Support — Venn Diagram" width="440"/>
</p>

Support is the most basic metric — the probability that both X and Y occur in the same transaction. It measures absolute frequency of co-occurrence.

- **Range:** 0 to 1
- **High support** → the pair appears frequently
- **Low support** → rare pair (but may still be interesting!)

---

### Confidence = P(Y|X)

**"When X is present, how often is Y also present?"**

<p align="center">
  <img src="docs/img/venn_confidence.svg" alt="Confidence — Venn Diagram" width="440"/>
</p>

Confidence is a **conditional probability**. It restricts the universe to only transactions containing X (the dashed circle), then asks: what fraction of those also contain Y?

- **Range:** 0 to 1
- **confidence = 0.60** → 60% of transactions with X also contain Y
- **Asymmetric:** conf(X→Y) ≠ conf(Y→X) in general

> **Pitfall:** High confidence alone doesn't mean X and Y are associated. If Y appears in 95% of all transactions, then conf(X→Y) ≈ 0.95 for *any* X. That's why we need **lift**.

---

### Lift = P(Y|X) / P(Y)

**"How much more likely is Y when X is present, compared to random chance?"**

<p align="center">
  <img src="docs/img/venn_lift.svg" alt="Lift — Bar Chart" width="440"/>
</p>

Lift compares the **observed** co-occurrence rate to the **expected** rate under independence. It corrects for popularity bias.

| Lift | Interpretation |
|------|---------------|
| **= 1.0** | X and Y are independent (no association) |
| **> 1.0** | Positive association — X makes Y more likely |
| **< 1.0** | Negative association — X makes Y less likely |
| **= 1.5** | Y is 50% more likely when X is present |

---

### Jaccard Similarity

**"What fraction of transactions containing X *or* Y contain both?"**

<p align="center">
  <img src="docs/img/venn_jaccard.svg" alt="Jaccard — Venn Diagram" width="440"/>
</p>

Jaccard focuses only on the **union** of X and Y — it completely ignores transactions where neither appears (¬a,¬b). This makes it useful when "absence" is not meaningful (e.g., in large catalogs).

- **Range:** 0 to 1
- **= 0** → no overlap
- **= 1** → identical sets
- **Symmetric:** jaccard(X,Y) = jaccard(Y,X)

**Related similarity metrics:** cosine, kulczynski, sokal_sneath, sokal_michener, rogers_tanimoto, hamming, lerman_similarity

---

### Leverage & Conviction

<p align="center">
  <img src="docs/img/leverage_conviction.svg" alt="Leverage and Conviction" width="720"/>
</p>

**Leverage** = P(X ∧ Y) − P(X)·P(Y) — the difference between observed and expected support. Ranges from −0.25 to +0.25. Zero means independence.

**Conviction** = (1 − P(Y)) / (1 − confidence) — measures the ratio of the expected error rate to the observed error rate. Values above 1 indicate a positive association; ∞ means the rule never fails.

---

## 3. All 60+ Metrics — Taxonomy

<p align="center">
  <img src="docs/img/metrics_taxonomy.svg" alt="Metrics Taxonomy" width="780"/>
</p>

<details>
<summary><b>Full list of all metrics with formulas</b></summary>

### Probabilistic
| Metric | Formula |
|--------|---------|
| `support` | a,b / n |
| `coverage` | a / n |
| `prevalence` | b / n |
| `confidence` | a,b / a |
| `reverse_confidence` | a,b / b |
| `lift` | confidence / prevalence |
| `leverage` | support − coverage × prevalence |
| `conviction` | (1 − prevalence) / (1 − confidence) |

### Similarity & Distance
| Metric | Formula |
|--------|---------|
| `jaccard` | a,b / (a + b − a,b) |
| `cosine` | support / √(coverage × prevalence) |
| `kulczynski` | 0.5 × (confidence + reverse_confidence) |
| `sokal_sneath` | a,b / (a + b − a,b + (a + b − 2·a,b)) |
| `sokal_michener` | (a,b + ¬a,¬b) / n |
| `rogers_tanimoto` | (a,b + ¬a,¬b) / (n + (a + b − 2·a,b)) |
| `hamming` | (a + b − 2·a,b) / n |

### Statistical Tests
| Metric | Formula |
|--------|---------|
| `chi_squared` | Pearson's χ² statistic |
| `phi_coefficient` | leverage / √(coverage × (1−coverage) × prevalence × (1−prevalence)) |
| `odds_ratio` | (a,b × ¬a,¬b) / (a,¬b × ¬a,b) with Haldane correction |
| `yules_q` | (OR − 1) / (OR + 1) |
| `yules_y` | (√OR − 1) / (√OR + 1) |
| `kappa` | Cohen's kappa coefficient |
| `p_value_approximation` | exp(−χ² / 2) |

### Information-Theoretic
| Metric | Formula |
|--------|---------|
| `mutual_information` | support × log₂(support / (coverage × prevalence)) |
| `j_measure` | support × log₂(conf / prev) + (coverage − support) × log₂((1−conf) / (1−prev)) |
| `weight_of_evidence` | log(P(X\|Y) / P(X\|¬Y)) with Laplace smoothing |
| `importance` | log₁₀(confidence / (1 − confidence)) |

### Interestingness
| Metric | Formula |
|--------|---------|
| `added_value` | confidence − prevalence |
| `certainty_factor` | (confidence − prevalence) / (1 − prevalence) |
| `zhangs_metric` | (confidence − P(Y\|¬X)) / max(confidence, P(Y\|¬X)) |
| `sebag_schoenauer` | confidence / (1 − confidence) |
| `relative_risk` | confidence / P(Y\|¬X) |
| `klosgen` | √support × added_value |
| `rule_power_factor` | support × confidence |
| `gini_index` | coverage × (conf² + (1−conf)²) + (1−coverage) × (P(Y\|¬X)² + P(¬Y\|¬X)²) − prev² − (1−prev)² |
| `collective_strength` | (support + (1−coverage−prevalence+support)) / (coverage×prevalence + (1−coverage)×(1−prevalence)) × ... |

### Smoothed & Confidence Intervals
| Metric | Formula |
|--------|---------|
| `confidence_laplace` | (a,b + 2) / (a + 4) |
| `confidence_lower` | confidence − 1.96 × SE |
| `confidence_upper` | confidence + 1.96 × SE |
| `odds_ratio_lower` | exp(ln(OR) − 1.96 × SE_log_OR) |
| `odds_ratio_upper` | exp(ln(OR) + 1.96 × SE_log_OR) |

</details>

---

## 4. Quick Start

### Direct Associations

When your data already has LHS/RHS columns (e.g., category→product, query→click):

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

### Combinatorial Associations

When each row is one item per transaction — all pairs generated automatically:

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

### From Pre-Aggregated Counts

When you already have the contingency table counts:

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

# Or compute only specific metrics:
result = association_measures(counts, measures=["support", "confidence", "lift"])
```

---

## 5. API Reference

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

---

## 6. Graph Algorithms

Build graphs from association metrics and discover structure:

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

---

## 7. Embedding & Visualization

Project items into 2D space based on their pairwise distances:

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

---

## 8. Algorithm Parameters

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

---

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
