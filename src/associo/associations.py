"""Direct and combinatorial association extraction using Polars."""

from __future__ import annotations

import polars as pl

from associo.measures import calculate_association_measures


def _aggregate_counts(
    lhs_rhs_distinct: pl.LazyFrame,
    lhs_distinct: pl.LazyFrame,
    rhs_distinct: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregate pair counts and compute the total transaction count.

    Returns a LazyFrame with columns:
        lhs, rhs, lhs_rhs_count, lhs_total_count, rhs_total_count, total_count
    """
    total_count = (
        pl.concat([
            lhs_distinct.select("tid"),
            rhs_distinct.select("tid"),
        ])
        .select(pl.col("tid").n_unique().alias("total_count"))
    )

    lhs_rhs_count = (
        lhs_rhs_distinct
        .group_by("lhs", "rhs")
        .agg(pl.col("tid").n_unique().alias("lhs_rhs_count"))
    )

    lhs_count = (
        lhs_distinct
        .group_by("lhs")
        .agg(pl.col("tid").n_unique().alias("lhs_total_count"))
    )

    rhs_count = (
        rhs_distinct
        .group_by("rhs")
        .agg(pl.col("tid").n_unique().alias("rhs_total_count"))
    )

    intersections = (
        lhs_count
        .join(rhs_count, how="cross")
        .join(total_count, how="cross")
        .join(lhs_rhs_count, on=["lhs", "rhs"], how="left")
        .with_columns(pl.col("lhs_rhs_count").fill_null(0))
    )

    return intersections


def compute_direct_associations(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_tid: str,
) -> pl.DataFrame:
    """Compute association measures when data already has lhs/rhs columns.

    Parameters
    ----------
    df : Source data with lhs, rhs, and transaction id columns.
    column_lhs : Name of the left-hand-side column.
    column_rhs : Name of the right-hand-side column.
    column_tid : Name of the transaction identifier column.

    Returns
    -------
    DataFrame with all association measures for each (lhs, rhs) pair.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    data = lf.select(
        pl.col(column_lhs).alias("lhs"),
        pl.col(column_rhs).alias("rhs"),
        pl.col(column_tid).alias("tid"),
    )

    lhs_distinct = data.select("tid", "lhs").unique()
    rhs_distinct = data.select("tid", "rhs").unique()
    lhs_rhs_distinct = data.select("tid", "lhs", "rhs").unique()

    counts = _aggregate_counts(lhs_rhs_distinct, lhs_distinct, rhs_distinct)

    return calculate_association_measures(counts)


def compute_combinatorial_associations(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_items: str,
    column_tid: str,
) -> pl.DataFrame:
    """Compute association measures for all item pairs sharing a transaction.

    Parameters
    ----------
    df : Source data with an item column and a transaction id column.
    column_items : Name of the items column.
    column_tid : Name of the transaction identifier column.

    Returns
    -------
    DataFrame with all association measures for every (lhs, rhs) combination.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    data = lf.select(
        pl.col(column_items).alias("item"),
        pl.col(column_tid).alias("tid"),
    )

    lhs_distinct = data.select("tid", pl.col("item").alias("lhs")).unique()
    rhs_distinct = data.select("tid", pl.col("item").alias("rhs")).unique()

    lhs_rhs_distinct = (
        lhs_distinct
        .join(rhs_distinct, on="tid")
        .select("tid", "lhs", "rhs")
        .unique()
    )

    counts = _aggregate_counts(lhs_rhs_distinct, lhs_distinct, rhs_distinct)

    return calculate_association_measures(counts)
