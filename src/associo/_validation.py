"""Input validation helpers."""

from __future__ import annotations

import polars as pl


def validate_columns(
    df: pl.DataFrame | pl.LazyFrame,
    required: list[str],
    *,
    func_name: str,
) -> None:
    """Raise a clear error if required columns are missing from the DataFrame."""
    if isinstance(df, pl.LazyFrame):
        existing = set(df.collect_schema().names())
    else:
        existing = set(df.columns)

    missing = [c for c in required if c not in existing]
    if missing:
        raise ValueError(
            f"{func_name}(): missing columns {missing} "
            f"in input DataFrame. Available: {sorted(existing)}"
        )
