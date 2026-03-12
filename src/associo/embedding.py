"""Dimensionality reduction (t-SNE) for visualization of item similarities."""

from __future__ import annotations

import polars as pl
import numpy as np
from sklearn.manifold import TSNE

from associo._validation import validate_columns


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

    # Symmetrise: take min distance for each pair (distance is symmetric)
    forward = df.select(
        pl.col(column_lhs).alias("a"),
        pl.col(column_rhs).alias("b"),
        pl.col(column_distance).alias("dist"),
    )
    reverse = df.select(
        pl.col(column_rhs).alias("a"),
        pl.col(column_lhs).alias("b"),
        pl.col(column_distance).alias("dist"),
    )
    sym = pl.concat([forward, reverse]).group_by("a", "b").agg(pl.col("dist").min()).sort("a", "b")

    all_items = sorted(set(sym["a"].to_list()) | set(sym["b"].to_list()))

    pivot = sym.pivot(on="b", index="a", values="dist").fill_null(1.0)
    # Ensure all items appear as columns and self-distance is 0
    for item in all_items:
        if item not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(1.0).alias(item))
    pivot = pivot.sort("a")

    items = pivot["a"].to_list()
    mat = pivot.select(items).to_numpy()
    # Ensure diagonal is 0 (self-distance)
    np.fill_diagonal(mat, 0.0)

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
