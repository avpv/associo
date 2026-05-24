"""Dense pairwise-matrix construction shared by matrix-based algorithms
(Affinity Propagation clustering, t-SNE embedding).

This is intentionally kept out of ``graph._build_graph``: a square dense
matrix is specific to matrix methods, not a concern of the generic graph core.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def pairwise_matrix(
    df: pl.DataFrame,
    col_a: str,
    col_b: str,
    col_value: str,
    *,
    fill_value: float,
    agg: str,
    zero_diagonal: bool = False,
) -> tuple[list, np.ndarray]:
    """Build a symmetric dense matrix from an edge list.

    Edges are symmetrised (each pair counted in both directions) and collapsed
    with ``agg`` ("min" or "max"). Missing entries are filled with
    ``fill_value``; rows and columns share the same sorted item order so the
    result is a proper square matrix.

    Returns ``(items, matrix)`` where ``items[i]`` labels row/column ``i``.
    """
    if agg not in ("min", "max"):
        raise ValueError(f"agg must be 'min' or 'max', got {agg!r}")

    forward = df.select(
        pl.col(col_a).alias("a"),
        pl.col(col_b).alias("b"),
        pl.col(col_value).alias("v"),
    )
    reverse = df.select(
        pl.col(col_b).alias("a"),
        pl.col(col_a).alias("b"),
        pl.col(col_value).alias("v"),
    )
    grouped = pl.concat([forward, reverse]).group_by("a", "b")
    value = pl.col("v")
    sym = grouped.agg(value.max() if agg == "max" else value.min()).sort("a", "b")

    all_items = sorted(set(sym["a"].to_list()) | set(sym["b"].to_list()))

    pivot = sym.pivot(on="b", index="a", values="v").fill_null(fill_value)
    for item in all_items:
        if item not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(float(fill_value)).alias(item))
    pivot = pivot.sort("a")

    items = pivot["a"].to_list()
    matrix = pivot.select(items).to_numpy()
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)

    return items, matrix
