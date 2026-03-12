"""Association measures computed entirely with Polars expressions.

Uses staged computation: base metrics are materialized first, then derived
metrics reference them via pl.col() to avoid redundant recomputation.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl


ROUND_PRECISION = 6

# Internal column names for contingency table cells used across stages.
_AB = "__ab"
_A = "__a"
_B = "__b"
_N = "__n"
_A_NOT_B = "__a_not_b"
_NOT_A_B = "__not_a_b"
_NOT_A_NOT_B = "__not_a_not_b"
_NOT_A_TOTAL = "__not_a_total"
_NOT_B_TOTAL = "__not_b_total"

# Columns used only as intermediate building blocks (cleaned up at the end).
_INTERNAL_COLS = frozenset({
    _AB, _A, _B, _N, _A_NOT_B, _NOT_A_B, _NOT_A_NOT_B, _NOT_A_TOTAL, _NOT_B_TOTAL,
    "__conf_lhs_to_not_rhs", "__conf_not_lhs_to_rhs",
})

# All available measure names for validation and selection.
ALL_MEASURES: frozenset[str] = frozenset({
    "lhs_not_rhs_count", "not_lhs_rhs_count", "not_lhs_not_rhs_count",
    "support", "coverage", "prevalence", "confidence", "reverse_confidence",
    "lift", "leverage", "confidence_laplace",
    "confidence_lower", "confidence_upper",
    "confidence_lower_laplace", "confidence_upper_laplace",
    "zhangs_metric", "importance", "importance_laplace",
    "added_value", "improvement", "cosine", "conviction",
    "jaccard", "kulczynski", "klosgen", "relative_difference",
    "rule_power_factor", "weight_of_evidence",
    "casual_confidence", "casual_support", "confirmed_confidence",
    "counter_example_rate", "implication_index", "lambda",
    "least_contradiction", "lerman_similarity", "lift_increase",
    "mutual_information", "relative_linkage_disequilibrium",
    "relative_risk", "standardized_lift", "sebag_schoenauer",
    "varying_rates_liaison", "support_vrl", "collective_strength",
    "confidence_boost", "odds_ratio", "odds_ratio_lower", "odds_ratio_upper",
    "certainty_factor", "imbalance_ratio", "gini_index",
    "hyper_confidence", "hyper_lift",
    "chi_squared", "p_value_approximation", "local_chi_squared",
    "phi_coefficient", "yules_q", "yules_y", "kappa", "j_measure",
    "difference_of_confidence", "hamming", "rogers_tanimoto",
    "sokal_michener", "sokal_sneath", "interestingness", "comprehensibility",
    "fisher_transformation_confidence", "fisher_transformation_reverse_confidence",
})


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Division that returns null when denominator is zero."""
    return pl.when(den != 0).then(num / den).otherwise(None)


def _laplace(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Laplace smoothing: (num + 2) / (den + 4)."""
    return _safe_div(num + 2, den + 4)


def _stage0_internals(
    col_lhs_rhs_count: str, col_lhs_total_count: str,
    col_rhs_total_count: str, col_total_count: str,
) -> list[pl.Expr]:
    """Cast input columns and compute contingency table cells."""
    ab = pl.col(col_lhs_rhs_count).cast(pl.Float64)
    a = pl.col(col_lhs_total_count).cast(pl.Float64)
    b = pl.col(col_rhs_total_count).cast(pl.Float64)
    n = pl.col(col_total_count).cast(pl.Float64)
    return [
        ab.alias(_AB),
        a.alias(_A),
        b.alias(_B),
        n.alias(_N),
        (a - ab).clip(lower_bound=0).alias(_A_NOT_B),
        (b - ab).clip(lower_bound=0).alias(_NOT_A_B),
        (n - a - b + ab).clip(lower_bound=0).alias(_NOT_A_NOT_B),
        (n - a).clip(lower_bound=0).alias(_NOT_A_TOTAL),
        (n - b).clip(lower_bound=0).alias(_NOT_B_TOTAL),
    ]


def _stage1_base() -> dict[str, pl.Expr]:
    """Base metrics computed from raw counts. Materialized before stage 2."""
    ab = pl.col(_AB)
    a = pl.col(_A)
    b = pl.col(_B)
    n = pl.col(_N)
    a_not_b = pl.col(_A_NOT_B)
    not_a_b = pl.col(_NOT_A_B)
    not_a_not_b = pl.col(_NOT_A_NOT_B)
    not_a_total = pl.col(_NOT_A_TOTAL)

    return {
        "lhs_not_rhs_count": a_not_b,
        "not_lhs_rhs_count": not_a_b,
        "not_lhs_not_rhs_count": not_a_not_b,
        "support": _safe_div(ab, n),
        "coverage": _safe_div(a, n),
        "prevalence": _safe_div(b, n),
        "confidence": _safe_div(ab, a),
        "reverse_confidence": _safe_div(ab, b),
        "confidence_laplace": _laplace(ab, a),
        # Directional confidences (needed by many stage-2 metrics)
        "__conf_lhs_to_not_rhs": _safe_div(a_not_b, a),
        "__conf_not_lhs_to_rhs": _safe_div(not_a_b, not_a_total),
    }


def _stage2_derived() -> dict[str, pl.Expr]:
    """Derived metrics that reference base metrics via pl.col()."""
    ab = pl.col(_AB)
    a = pl.col(_A)
    b = pl.col(_B)
    n = pl.col(_N)
    a_not_b = pl.col(_A_NOT_B)
    not_a_b = pl.col(_NOT_A_B)
    not_a_not_b = pl.col(_NOT_A_NOT_B)
    not_a_total = pl.col(_NOT_A_TOTAL)
    not_b_total = pl.col(_NOT_B_TOTAL)

    support = pl.col("support")
    coverage = pl.col("coverage")
    prevalence = pl.col("prevalence")
    confidence = pl.col("confidence")
    reverse_confidence = pl.col("reverse_confidence")
    confidence_laplace = pl.col("confidence_laplace")
    conf_lhs_to_not_rhs = pl.col("__conf_lhs_to_not_rhs")
    conf_not_lhs_to_rhs = pl.col("__conf_not_lhs_to_rhs")

    lift = _safe_div(confidence, prevalence)
    improvement = confidence - conf_not_lhs_to_rhs

    z = 1.96
    se = (confidence * (1.0 - confidence) / a).sqrt()
    se_lap = (confidence_laplace * (1.0 - confidence_laplace) / a).sqrt()

    # Odds ratio with Haldane correction
    ah = ab + 0.5
    bh = a_not_b + 0.5
    ch = not_a_b + 0.5
    dh = not_a_not_b + 0.5
    odds_ratio = _safe_div(ah * dh, bh * ch)
    log_or = odds_ratio.log()
    se_or = (1.0 / ah + 1.0 / bh + 1.0 / ch + 1.0 / dh).sqrt()

    # Chi-squared components
    exp_ab = coverage * prevalence * n
    exp_a_not_b = coverage * (1.0 - prevalence) * n
    exp_not_a_b = (1.0 - coverage) * prevalence * n
    exp_not_a_not_b = (1.0 - coverage) * (1.0 - prevalence) * n
    chi_sq = (
        pl.when(exp_ab != 0).then((ab - exp_ab).pow(2) / exp_ab).otherwise(0)
        + pl.when(exp_a_not_b != 0).then((a_not_b - exp_a_not_b).pow(2) / exp_a_not_b).otherwise(0)
        + pl.when(exp_not_a_b != 0).then((not_a_b - exp_not_a_b).pow(2) / exp_not_a_b).otherwise(0)
        + pl.when(exp_not_a_not_b != 0).then((not_a_not_b - exp_not_a_not_b).pow(2) / exp_not_a_not_b).otherwise(0)
    )

    # Relative linkage disequilibrium
    d_val = ab * not_a_not_b - a_not_b * not_a_b
    min_pos = pl.min_horizontal(a_not_b, not_a_b)
    min_neg = pl.min_horizontal(ab, not_a_not_b)

    support_union = coverage + prevalence - support
    support_lhs_not_rhs = coverage - support

    complementary_conf = _safe_div(not_a_not_b, not_a_total)

    return {
        "lift": lift,
        "leverage": support - coverage * prevalence,
        "confidence_lower": confidence - z * se,
        "confidence_upper": confidence + z * se,
        "confidence_lower_laplace": confidence_laplace - z * se_lap,
        "confidence_upper_laplace": confidence_laplace + z * se_lap,
        "zhangs_metric": _safe_div(
            confidence - conf_not_lhs_to_rhs,
            pl.max_horizontal(confidence, conf_not_lhs_to_rhs),
        ),
        "importance": _safe_div(confidence, conf_lhs_to_not_rhs).log(base=10),
        "importance_laplace": (_laplace(ab, a) / _laplace(a_not_b, a)).log(base=10),
        "added_value": confidence - prevalence,
        "improvement": improvement,
        "cosine": _safe_div(support, (coverage * prevalence).sqrt()),
        "conviction": _safe_div(1.0 - prevalence, 1.0 - confidence),
        "jaccard": _safe_div(support, coverage + prevalence - support),
        "kulczynski": 0.5 * (_safe_div(support, coverage) + _safe_div(support, prevalence)),
        "klosgen": support.sqrt() * (confidence - prevalence),
        "relative_difference": _safe_div(confidence - prevalence, prevalence),
        "rule_power_factor": support * confidence,
        "weight_of_evidence": _laplace(ab, b).log() - _laplace(a_not_b, not_b_total).log(),
        "casual_confidence": 0.5 * (confidence + complementary_conf),
        "casual_support": support_union + (1.0 - support),
        "confirmed_confidence": confidence - conf_lhs_to_not_rhs,
        "counter_example_rate": _safe_div(ab + not_a_b, n),
        "implication_index": _safe_div(support - coverage * prevalence, (coverage * prevalence).sqrt()),
        "lambda": _safe_div(
            1.0 - pl.max_horizontal(prevalence, 1.0 - prevalence) - (1.0 - pl.max_horizontal(confidence, conf_not_lhs_to_rhs)),
            1.0 - pl.max_horizontal(prevalence, 1.0 - prevalence),
        ),
        "least_contradiction": _safe_div(support_union - ((1.0 - prevalence) + support), prevalence),
        "lerman_similarity": _safe_div(support_union - coverage * prevalence, (coverage * prevalence).sqrt()),
        "lift_increase": _safe_div(lift - 1.0, prevalence),
        "mutual_information": support * (support / (coverage * prevalence)).log(base=2),
        "relative_linkage_disequilibrium": pl.when(d_val > 0).then(
            _safe_div(d_val, d_val + min_pos)
        ).otherwise(
            _safe_div(d_val, d_val - min_neg)
        ),
        "relative_risk": _safe_div(confidence, conf_not_lhs_to_rhs),
        "standardized_lift": _safe_div(lift - 1.0, lift + 1.0),
        "sebag_schoenauer": _safe_div(confidence, conf_lhs_to_not_rhs),
        "varying_rates_liaison": lift - 1.0,
        "support_vrl": support * (lift - 1.0),
        "collective_strength": _safe_div(
            support * (1.0 - support),
            (coverage * prevalence - support) * (coverage + prevalence - support),
        ),
        "confidence_boost": _safe_div(confidence, confidence - improvement),
        "odds_ratio": odds_ratio,
        "odds_ratio_lower": (log_or - z * se_or).exp(),
        "odds_ratio_upper": (log_or + z * se_or).exp(),
        "certainty_factor": _safe_div(confidence - prevalence, 1.0 - prevalence),
        "imbalance_ratio": _safe_div(coverage - prevalence, coverage + prevalence - support),
        "gini_index": support * (1.0 - confidence.pow(2) - (1.0 - confidence).pow(2)),
        "hyper_confidence": _safe_div(confidence, conf_lhs_to_not_rhs),
        "hyper_lift": _safe_div(confidence, conf_not_lhs_to_rhs),
        "chi_squared": chi_sq,
        "p_value_approximation": (-chi_sq / 2).exp(),
        "local_chi_squared": _safe_div((support * n - exp_ab).pow(2), exp_ab),
        "phi_coefficient": _safe_div(
            support - coverage * prevalence,
            (coverage * (1.0 - coverage) * prevalence * (1.0 - prevalence)).sqrt(),
        ),
        "yules_q": _safe_div(odds_ratio - 1.0, odds_ratio + 1.0),
        "yules_y": _safe_div(odds_ratio.sqrt() - 1.0, odds_ratio.sqrt() + 1.0),
        "kappa": _safe_div(
            (support + 1.0 - coverage - prevalence + support)
            - (coverage * prevalence + (1.0 - coverage) * (1.0 - prevalence)),
            1.0 - (coverage * prevalence + (1.0 - coverage) * (1.0 - prevalence)),
        ),
        "j_measure": (
            support * (confidence / prevalence).log(base=2)
            + support_lhs_not_rhs * (conf_lhs_to_not_rhs / (1.0 - prevalence)).log(base=2)
        ),
        "difference_of_confidence": confidence - conf_not_lhs_to_rhs,
        "hamming": _safe_div(a + b - 2.0 * ab, n),
        "rogers_tanimoto": _safe_div(a + b - 2.0 * ab + n, a + b + n - ab),
        "sokal_michener": _safe_div(ab + not_a_not_b, n),
        "sokal_sneath": _safe_div(2.0 * ab, a + b),
        "interestingness": confidence * reverse_confidence * (1.0 - confidence),
        "comprehensibility": _safe_div((1.0 + prevalence).log(), (1.0 + support).log()),
        "fisher_transformation_confidence": 2.0 * confidence.sqrt().arcsin(),
        "fisher_transformation_reverse_confidence": 2.0 * reverse_confidence.sqrt().arcsin(),
    }


def calculate_association_measures(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    col_lhs_rhs_count: str = "lhs_rhs_count",
    col_lhs_total_count: str = "lhs_total_count",
    col_rhs_total_count: str = "rhs_total_count",
    col_total_count: str = "total_count",
    measures: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Calculate association measures from pre-aggregated counts.

    Uses staged computation: base metrics (support, confidence, etc.) are
    materialized first, then derived metrics reference them via pl.col().
    This avoids redundant recomputation of shared sub-expressions.

    Parameters
    ----------
    df : DataFrame/LazyFrame with columns for the four counts.
    col_lhs_rhs_count : Column with count(lhs ∩ rhs).
    col_lhs_total_count : Column with count(lhs).
    col_rhs_total_count : Column with count(rhs).
    col_total_count : Column with total transaction count.
    measures : Subset of measure names to compute. None = all measures.
        Use ``associo.measures.ALL_MEASURES`` to see available names.

    Returns
    -------
    DataFrame with selected association measure columns appended.
    """
    from associo._validation import validate_columns

    if measures is not None:
        requested = set(measures)
        unknown = requested - ALL_MEASURES
        if unknown:
            raise ValueError(f"Unknown measures: {unknown}. Available: {sorted(ALL_MEASURES)}")
    else:
        requested = None  # means "all"

    validate_columns(
        df,
        [col_lhs_rhs_count, col_lhs_total_count, col_rhs_total_count, col_total_count],
        func_name="calculate_association_measures",
    )

    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Stage 0: cast inputs + contingency table cells
    lf = lf.with_columns(_stage0_internals(
        col_lhs_rhs_count, col_lhs_total_count, col_rhs_total_count, col_total_count,
    ))

    # Stage 1: base metrics (support, confidence, etc.) — materialized
    stage1 = _stage1_base()
    # Always compute all base metrics (they're cheap and needed by stage 2)
    lf = lf.with_columns([expr.alias(name) for name, expr in stage1.items()])

    # Stage 2: derived metrics that reference base via pl.col()
    stage2 = _stage2_derived()

    if requested is not None:
        # Only compute requested derived metrics (stage1 metrics already present)
        needed_stage2 = {k: v for k, v in stage2.items() if k in requested}
    else:
        needed_stage2 = stage2

    if needed_stage2:
        lf = lf.with_columns([
            expr.round(ROUND_PRECISION).alias(name)
            for name, expr in needed_stage2.items()
        ])

    # Round stage-1 metrics that are in output
    stage1_public = {k for k in stage1 if not k.startswith("__")}
    if requested is not None:
        stage1_to_round = stage1_public & requested
    else:
        stage1_to_round = stage1_public

    if stage1_to_round:
        lf = lf.with_columns([
            pl.col(name).round(ROUND_PRECISION) for name in stage1_to_round
        ])

    # Determine final columns: original + requested measures
    original_cols = (df.columns if isinstance(df, pl.DataFrame) else df.collect_schema().names())
    if requested is not None:
        output_measures = [m for m in (*stage1.keys(), *stage2.keys())
                          if m in requested and not m.startswith("__")]
    else:
        output_measures = [m for m in (*stage1.keys(), *stage2.keys()) if not m.startswith("__")]

    return lf.select(original_cols + output_measures).collect()
