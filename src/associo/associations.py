"""Direct and combinatorial association extraction using Polars."""

from __future__ import annotations

import polars as pl

from associo._validation import validate_columns
from associo.measures import calculate_association_measures


def _aggregate_counts(
    lhs_rhs_distinct: pl.LazyFrame,
    lhs_distinct: pl.LazyFrame,
    rhs_distinct: pl.LazyFrame,
    *,
    all_pairs: bool = False,
) -> pl.LazyFrame:
    """Aggregate pair counts and compute the total transaction count.

    Parameters
    ----------
    all_pairs : If True, compute measures for ALL (lhs × rhs) combinations,
        including pairs that never co-occur (lhs_rhs_count=0).
        If False (default), only compute for actually co-occurring pairs.

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

    if all_pairs:
        # Cross join: all (lhs × rhs) including non-co-occurring pairs
        result = (
            lhs_count
            .join(rhs_count, how="cross")
            .join(total_count, how="cross")
            .join(lhs_rhs_count, on=["lhs", "rhs"], how="left")
            .with_columns(pl.col("lhs_rhs_count").fill_null(0))
        )
    else:
        # Only actually co-occurring pairs — no cross join explosion
        result = (
            lhs_rhs_count
            .join(lhs_count, on="lhs")
            .join(rhs_count, on="rhs")
            .join(total_count, how="cross")
        )

    return result


def compute_direct_associations(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_lhs: str,
    column_rhs: str,
    column_tid: str,
    all_pairs: bool = False,
) -> pl.DataFrame:
    """Compute association measures when data already has lhs/rhs columns.

    Parameters
    ----------
    df : Source data with lhs, rhs, and transaction id columns.
    column_lhs : Name of the left-hand-side column.
    column_rhs : Name of the right-hand-side column.
    column_tid : Name of the transaction identifier column.
    all_pairs : If True, compute for all (lhs × rhs) combinations including
        non-co-occurring pairs. Default False — only co-occurring pairs.

    Returns
    -------
    DataFrame with all association measures for each (lhs, rhs) pair.
    """
    validate_columns(df, [column_lhs, column_rhs, column_tid], func_name="compute_direct_associations")

    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    data = lf.select(
        pl.col(column_lhs).alias("lhs"),
        pl.col(column_rhs).alias("rhs"),
        pl.col(column_tid).alias("tid"),
    )

    lhs_distinct = data.select("tid", "lhs").unique()
    rhs_distinct = data.select("tid", "rhs").unique()
    lhs_rhs_distinct = data.select("tid", "lhs", "rhs").unique()

    counts = _aggregate_counts(lhs_rhs_distinct, lhs_distinct, rhs_distinct, all_pairs=all_pairs)

    return calculate_association_measures(counts)


def compute_combinatorial_associations(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column_items: str,
    column_tid: str,
    include_self_pairs: bool = False,
    all_pairs: bool = False,
) -> pl.DataFrame:
    """Compute association measures for all item pairs sharing a transaction.

    Parameters
    ----------
    df : Source data with an item column and a transaction id column.
    column_items : Name of the items column.
    column_tid : Name of the transaction identifier column.
    include_self_pairs : If True, include pairs where lhs == rhs (A→A).
        Default False.
    all_pairs : If True, compute for all (item × item) combinations including
        non-co-occurring pairs. Default False.

    Returns
    -------
    DataFrame with all association measures for every (lhs, rhs) combination.
    """
    validate_columns(df, [column_items, column_tid], func_name="compute_combinatorial_associations")

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

    if not include_self_pairs:
        lhs_rhs_distinct = lhs_rhs_distinct.filter(pl.col("lhs") != pl.col("rhs"))

    counts = _aggregate_counts(lhs_rhs_distinct, lhs_distinct, rhs_distinct, all_pairs=all_pairs)

    return calculate_association_measures(counts)
