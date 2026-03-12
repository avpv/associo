"""Association measures computed entirely with Polars expressions."""

from __future__ import annotations

from typing import Sequence

import polars as pl


ROUND_PRECISION = 6


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Division that returns null when denominator is zero."""
    return pl.when(den != 0).then(num / den).otherwise(None)


def _laplace(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Laplace smoothing: (num + 2) / (den + 4)."""
    return _safe_div(num + 2, den + 4)


# All available measure names for validation and selection
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


def _build_all_measures(
    ab: pl.Expr, a: pl.Expr, b: pl.Expr, n: pl.Expr,
) -> dict[str, pl.Expr]:
    """Build a dict of measure_name → Polars expression."""

    # Contingency table cells
    a_not_b = (a - ab).clip(lower_bound=0)
    not_a_b = (b - ab).clip(lower_bound=0)
    not_a_not_b = (n - a - b + ab).clip(lower_bound=0)
    not_a_total = (n - a).clip(lower_bound=0)
    not_b_total = (n - b).clip(lower_bound=0)

    # Basic
    support = _safe_div(ab, n)
    coverage = _safe_div(a, n)
    prevalence = _safe_div(b, n)
    confidence = _safe_div(ab, a)
    reverse_confidence = _safe_div(ab, b)
    lift = _safe_div(confidence, prevalence)
    leverage = support - coverage * prevalence

    confidence_laplace = _laplace(ab, a)

    z = 1.96
    se = (confidence * (pl.lit(1.0) - confidence) / a).sqrt()
    confidence_lower = confidence - z * se
    confidence_upper = confidence + z * se

    se_lap = (confidence_laplace * (pl.lit(1.0) - confidence_laplace) / a).sqrt()
    confidence_lower_laplace = confidence_laplace - z * se_lap
    confidence_upper_laplace = confidence_laplace + z * se_lap

    conf_lhs_to_not_rhs = _safe_div(a_not_b, a)
    conf_not_lhs_to_rhs = _safe_div(not_a_b, not_a_total)

    max_conf = pl.max_horizontal(confidence, conf_not_lhs_to_rhs)
    zhangs_metric = _safe_div(confidence - conf_not_lhs_to_rhs, max_conf)

    importance = _safe_div(confidence, conf_lhs_to_not_rhs).log(base=10)
    importance_laplace = (_laplace(ab, a) / _laplace(a_not_b, a)).log(base=10)

    added_value = confidence - prevalence
    improvement = confidence - conf_not_lhs_to_rhs
    cosine = _safe_div(support, (coverage * prevalence).sqrt())
    conviction = _safe_div(pl.lit(1.0) - prevalence, pl.lit(1.0) - confidence)
    jaccard = _safe_div(support, coverage + prevalence - support)
    kulczynski = pl.lit(0.5) * (_safe_div(support, coverage) + _safe_div(support, prevalence))
    klosgen = support.sqrt() * (confidence - prevalence)
    relative_difference = _safe_div(confidence - prevalence, prevalence)
    rule_power_factor = support * confidence

    woe_num = _laplace(ab, b)
    woe_den = _laplace(a_not_b, not_b_total)
    weight_of_evidence = woe_num.log() - woe_den.log()

    complementary_conf = _safe_div(not_a_not_b, not_a_total)
    casual_confidence = pl.lit(0.5) * (confidence + complementary_conf)

    support_union = coverage + prevalence - support
    casual_support = support_union + (pl.lit(1.0) - support)

    confirmed_confidence = confidence - conf_lhs_to_not_rhs
    counter_example_rate = _safe_div(ab + not_a_b, n)
    implication_index = _safe_div(support - coverage * prevalence, (coverage * prevalence).sqrt())

    baseline_error = pl.lit(1.0) - pl.max_horizontal(prevalence, pl.lit(1.0) - prevalence)
    conditional_error = pl.lit(1.0) - pl.max_horizontal(confidence, conf_not_lhs_to_rhs)
    lambda_measure = _safe_div(baseline_error - conditional_error, baseline_error)

    support_union_not_y = (pl.lit(1.0) - prevalence) + support
    least_contradiction = _safe_div(support_union - support_union_not_y, prevalence)

    lerman_similarity = _safe_div(support_union - coverage * prevalence, (coverage * prevalence).sqrt())

    lift_increase = _safe_div(lift - pl.lit(1.0), prevalence)
    mutual_information = support * (support / (coverage * prevalence)).log(base=2)

    d_val = ab * not_a_not_b - a_not_b * not_a_b
    min_pos = pl.min_horizontal(a_not_b, not_a_b)
    min_neg = pl.min_horizontal(ab, not_a_not_b)
    rld = pl.when(d_val > 0).then(_safe_div(d_val, d_val + min_pos)).otherwise(_safe_div(d_val, d_val - min_neg))

    relative_risk = _safe_div(confidence, conf_not_lhs_to_rhs)
    standardized_lift = _safe_div(lift - pl.lit(1.0), lift + pl.lit(1.0))
    sebag_schoenauer = _safe_div(confidence, conf_lhs_to_not_rhs)
    varying_rates_liaison = lift - pl.lit(1.0)
    support_vrl = support * (lift - pl.lit(1.0))

    collective_strength = _safe_div(
        support * (pl.lit(1.0) - support),
        (coverage * prevalence - support) * (coverage + prevalence - support),
    )

    confidence_boost = _safe_div(confidence, confidence - improvement)

    ah = ab + 0.5
    bh = a_not_b + 0.5
    ch = not_a_b + 0.5
    dh = not_a_not_b + 0.5
    odds_ratio = _safe_div(ah * dh, bh * ch)

    log_or = odds_ratio.log()
    se_or = (pl.lit(1.0) / ah + pl.lit(1.0) / bh + pl.lit(1.0) / ch + pl.lit(1.0) / dh).sqrt()
    odds_ratio_lower = (log_or - z * se_or).exp()
    odds_ratio_upper = (log_or + z * se_or).exp()

    certainty_factor = _safe_div(confidence - prevalence, pl.lit(1.0) - prevalence)
    imbalance_ratio = _safe_div(coverage - prevalence, coverage + prevalence - support)
    gini_index = support * (pl.lit(1.0) - confidence.pow(2) - (pl.lit(1.0) - confidence).pow(2))
    hyper_confidence = _safe_div(confidence, conf_lhs_to_not_rhs)
    hyper_lift = _safe_div(confidence, conf_not_lhs_to_rhs)

    exp_ab = coverage * prevalence * n
    exp_a_not_b = coverage * (pl.lit(1.0) - prevalence) * n
    exp_not_a_b = (pl.lit(1.0) - coverage) * prevalence * n
    exp_not_a_not_b = (pl.lit(1.0) - coverage) * (pl.lit(1.0) - prevalence) * n

    chi_sq = (
        pl.when(exp_ab != 0).then((ab - exp_ab).pow(2) / exp_ab).otherwise(0)
        + pl.when(exp_a_not_b != 0).then((a_not_b - exp_a_not_b).pow(2) / exp_a_not_b).otherwise(0)
        + pl.when(exp_not_a_b != 0).then((not_a_b - exp_not_a_b).pow(2) / exp_not_a_b).otherwise(0)
        + pl.when(exp_not_a_not_b != 0)
        .then((not_a_not_b - exp_not_a_not_b).pow(2) / exp_not_a_not_b)
        .otherwise(0)
    )

    p_value_approx = (-chi_sq / 2).exp()
    local_chi_sq = _safe_div((support * n - exp_ab).pow(2), exp_ab)

    phi_coefficient = _safe_div(
        support - coverage * prevalence,
        (coverage * (pl.lit(1.0) - coverage) * prevalence * (pl.lit(1.0) - prevalence)).sqrt(),
    )

    yules_q = _safe_div(odds_ratio - pl.lit(1.0), odds_ratio + pl.lit(1.0))
    yules_y = _safe_div(odds_ratio.sqrt() - pl.lit(1.0), odds_ratio.sqrt() + pl.lit(1.0))

    observed_agreement = support + (pl.lit(1.0) - coverage - prevalence + support)
    expected_agreement = coverage * prevalence + (pl.lit(1.0) - coverage) * (pl.lit(1.0) - prevalence)
    kappa = _safe_div(observed_agreement - expected_agreement, pl.lit(1.0) - expected_agreement)

    support_lhs_not_rhs = coverage - support
    j_rhs = support * (confidence / prevalence).log(base=2)
    j_not_rhs = support_lhs_not_rhs * (conf_lhs_to_not_rhs / (pl.lit(1.0) - prevalence)).log(base=2)
    j_measure = j_rhs + j_not_rhs

    difference_of_confidence = confidence - conf_not_lhs_to_rhs

    hamming = _safe_div(a + b - pl.lit(2.0) * ab, n)
    rogers_tanimoto = _safe_div(a + b - pl.lit(2.0) * ab + n, a + b + n - ab)
    sokal_michener = _safe_div(ab + not_a_not_b, n)
    sokal_sneath = _safe_div(pl.lit(2.0) * ab, a + b)
    interestingness = confidence * reverse_confidence * (pl.lit(1.0) - confidence)
    comprehensibility = _safe_div((pl.lit(1.0) + prevalence).log(), (pl.lit(1.0) + support).log())

    fisher_conf = pl.lit(2.0) * confidence.sqrt().arcsin()
    fisher_rev_conf = pl.lit(2.0) * reverse_confidence.sqrt().arcsin()

    return {
        "lhs_not_rhs_count": a_not_b,
        "not_lhs_rhs_count": not_a_b,
        "not_lhs_not_rhs_count": not_a_not_b,
        "support": support,
        "coverage": coverage,
        "prevalence": prevalence,
        "confidence": confidence,
        "reverse_confidence": reverse_confidence,
        "lift": lift,
        "leverage": leverage,
        "confidence_laplace": confidence_laplace,
        "confidence_lower": confidence_lower,
        "confidence_upper": confidence_upper,
        "confidence_lower_laplace": confidence_lower_laplace,
        "confidence_upper_laplace": confidence_upper_laplace,
        "zhangs_metric": zhangs_metric,
        "importance": importance,
        "importance_laplace": importance_laplace,
        "added_value": added_value,
        "improvement": improvement,
        "cosine": cosine,
        "conviction": conviction,
        "jaccard": jaccard,
        "kulczynski": kulczynski,
        "klosgen": klosgen,
        "relative_difference": relative_difference,
        "rule_power_factor": rule_power_factor,
        "weight_of_evidence": weight_of_evidence,
        "casual_confidence": casual_confidence,
        "casual_support": casual_support,
        "confirmed_confidence": confirmed_confidence,
        "counter_example_rate": counter_example_rate,
        "implication_index": implication_index,
        "lambda": lambda_measure,
        "least_contradiction": least_contradiction,
        "lerman_similarity": lerman_similarity,
        "lift_increase": lift_increase,
        "mutual_information": mutual_information,
        "relative_linkage_disequilibrium": rld,
        "relative_risk": relative_risk,
        "standardized_lift": standardized_lift,
        "sebag_schoenauer": sebag_schoenauer,
        "varying_rates_liaison": varying_rates_liaison,
        "support_vrl": support_vrl,
        "collective_strength": collective_strength,
        "confidence_boost": confidence_boost,
        "odds_ratio": odds_ratio,
        "odds_ratio_lower": odds_ratio_lower,
        "odds_ratio_upper": odds_ratio_upper,
        "certainty_factor": certainty_factor,
        "imbalance_ratio": imbalance_ratio,
        "gini_index": gini_index,
        "hyper_confidence": hyper_confidence,
        "hyper_lift": hyper_lift,
        "chi_squared": chi_sq,
        "p_value_approximation": p_value_approx,
        "local_chi_squared": local_chi_sq,
        "phi_coefficient": phi_coefficient,
        "yules_q": yules_q,
        "yules_y": yules_y,
        "kappa": kappa,
        "j_measure": j_measure,
        "difference_of_confidence": difference_of_confidence,
        "hamming": hamming,
        "rogers_tanimoto": rogers_tanimoto,
        "sokal_michener": sokal_michener,
        "sokal_sneath": sokal_sneath,
        "interestingness": interestingness,
        "comprehensibility": comprehensibility,
        "fisher_transformation_confidence": fisher_conf,
        "fisher_transformation_reverse_confidence": fisher_rev_conf,
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
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    ab = pl.col(col_lhs_rhs_count).cast(pl.Float64)
    a = pl.col(col_lhs_total_count).cast(pl.Float64)
    b = pl.col(col_rhs_total_count).cast(pl.Float64)
    n = pl.col(col_total_count).cast(pl.Float64)

    all_exprs = _build_all_measures(ab, a, b, n)

    if measures is not None:
        unknown = set(measures) - ALL_MEASURES
        if unknown:
            raise ValueError(f"Unknown measures: {unknown}. Available: {sorted(ALL_MEASURES)}")
        selected = {k: v for k, v in all_exprs.items() if k in set(measures)}
    else:
        selected = all_exprs

    columns = [expr.round(ROUND_PRECISION).alias(name) for name, expr in selected.items()]
    result = lf.with_columns(columns)

    return result.collect()
