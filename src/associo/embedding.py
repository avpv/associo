"""Dimensionality reduction (t-SNE) for visualization of item similarities."""

from __future__ import annotations

import polars as pl
import numpy as np
from sklearn.manifold import TSNE


def compute_embedding(
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
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    pivot = df.pivot(
        on=column_rhs,
        index=column_lhs,
        values=column_distance,
    ).fill_null(1.0)

    items = pivot[column_lhs].to_list()
    mat = pivot.drop(column_lhs).to_numpy()

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
