"""Association measures computed entirely with Polars expressions."""

from __future__ import annotations

import polars as pl
import math


ROUND_PRECISION = 6


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Division that returns null when denominator is zero."""
    return pl.when(den != 0).then(num / den).otherwise(None)


def _laplace(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Laplace smoothing: (num + 2) / (den + 4)."""
    return _safe_div(num + 2, den + 4)


def calculate_association_measures(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    col_lhs_rhs_count: str = "lhs_rhs_count",
    col_lhs_total_count: str = "lhs_total_count",
    col_rhs_total_count: str = "rhs_total_count",
    col_total_count: str = "total_count",
) -> pl.DataFrame:
    """Calculate the full set of association measures from pre-aggregated counts.

    Parameters
    ----------
    df : DataFrame/LazyFrame with columns for the four counts.
    col_lhs_rhs_count : Column with count(lhs ∩ rhs).
    col_lhs_total_count : Column with count(lhs).
    col_rhs_total_count : Column with count(rhs).
    col_total_count : Column with total transaction count.

    Returns
    -------
    DataFrame with all association measure columns appended.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Aliases for readability
    ab = pl.col(col_lhs_rhs_count).cast(pl.Float64)
    a = pl.col(col_lhs_total_count).cast(pl.Float64)
    b = pl.col(col_rhs_total_count).cast(pl.Float64)
    n = pl.col(col_total_count).cast(pl.Float64)

    # Contingency table cells
    a_not_b = (a - ab).clip(lower_bound=0)
    not_a_b = (b - ab).clip(lower_bound=0)
    not_a_not_b = (n - a - b + ab).clip(lower_bound=0)
    not_a_total = (n - a).clip(lower_bound=0)
    not_b_total = (n - b).clip(lower_bound=0)

    # Basic measures
    support = _safe_div(ab, n)
    coverage = _safe_div(a, n)
    prevalence = _safe_div(b, n)
    confidence = _safe_div(ab, a)
    reverse_confidence = _safe_div(ab, b)
    lift = _safe_div(confidence, prevalence)
    leverage = support - coverage * prevalence

    # Laplace smoothed confidence
    confidence_laplace = _laplace(ab, a)

    # Confidence intervals (z=1.96)
    z = 1.96
    se = (confidence * (pl.lit(1.0) - confidence) / a).sqrt()
    confidence_lower = confidence - z * se
    confidence_upper = confidence + z * se

    se_lap = (confidence_laplace * (pl.lit(1.0) - confidence_laplace) / a).sqrt()
    confidence_lower_laplace = confidence_laplace - z * se_lap
    confidence_upper_laplace = confidence_laplace + z * se_lap

    # Directional confidences
    conf_lhs_to_not_rhs = _safe_div(a_not_b, a)
    conf_not_lhs_to_rhs = _safe_div(not_a_b, not_a_total)

    # Zhang's metric
    max_conf = pl.max_horizontal(confidence, conf_not_lhs_to_rhs)
    zhangs_metric = _safe_div(confidence - conf_not_lhs_to_rhs, max_conf)

    # Importance (log10)
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

    # Weight of evidence
    woe_num = _laplace(ab, b)
    woe_den = _laplace(a_not_b, not_b_total)
    weight_of_evidence = woe_num.log() - woe_den.log()

    # Casual confidence
    complementary_conf = _safe_div(not_a_not_b, not_a_total)
    casual_confidence = pl.lit(0.5) * (confidence + complementary_conf)

    # Casual support
    support_union = coverage + prevalence - support
    casual_support = support_union + (pl.lit(1.0) - support)

    confirmed_confidence = confidence - conf_lhs_to_not_rhs
    counter_example_rate = _safe_div(ab + not_a_b, n)
    implication_index = _safe_div(support - coverage * prevalence, (coverage * prevalence).sqrt())

    # Lambda
    baseline_error = pl.lit(1.0) - pl.max_horizontal(prevalence, pl.lit(1.0) - prevalence)
    conditional_error = pl.lit(1.0) - pl.max_horizontal(confidence, conf_not_lhs_to_rhs)
    lambda_measure = _safe_div(baseline_error - conditional_error, baseline_error)

    # Least contradiction
    support_union_not_y = (pl.lit(1.0) - prevalence) + support
    least_contradiction = _safe_div(support_union - support_union_not_y, prevalence)

    # Lerman similarity
    lerman_similarity = _safe_div(support_union - coverage * prevalence, (coverage * prevalence).sqrt())

    lift_increase = _safe_div(lift - pl.lit(1.0), prevalence)
    mutual_information = support * (support / (coverage * prevalence)).log(base=2)

    # Relative linkage disequilibrium
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

    # Odds ratio with Haldane correction (+0.5)
    ah = ab + 0.5
    bh = a_not_b + 0.5
    ch = not_a_b + 0.5
    dh = not_a_not_b + 0.5
    odds_ratio = _safe_div(ah * dh, bh * ch)

    # Odds ratio CI
    log_or = odds_ratio.log()
    se_or = (
        pl.lit(1.0) / ah + pl.lit(1.0) / bh + pl.lit(1.0) / ch + pl.lit(1.0) / dh
    ).sqrt()
    odds_ratio_lower = (log_or - z * se_or).exp()
    odds_ratio_upper = (log_or + z * se_or).exp()

    certainty_factor = _safe_div(confidence - prevalence, pl.lit(1.0) - prevalence)
    imbalance_ratio = _safe_div(coverage - prevalence, coverage + prevalence - support)
    gini_index = support * (pl.lit(1.0) - confidence.pow(2) - (pl.lit(1.0) - confidence).pow(2))
    hyper_confidence = _safe_div(confidence, conf_lhs_to_not_rhs)
    hyper_lift = _safe_div(confidence, conf_not_lhs_to_rhs)

    # Chi-squared
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

    # J-measure
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

    result = lf.with_columns(
        # Contingency cells
        a_not_b.round(ROUND_PRECISION).alias("lhs_not_rhs_count"),
        not_a_b.round(ROUND_PRECISION).alias("not_lhs_rhs_count"),
        not_a_not_b.round(ROUND_PRECISION).alias("not_lhs_not_rhs_count"),
        # Measures
        support.round(ROUND_PRECISION).alias("support"),
        coverage.round(ROUND_PRECISION).alias("coverage"),
        prevalence.round(ROUND_PRECISION).alias("prevalence"),
        confidence.round(ROUND_PRECISION).alias("confidence"),
        reverse_confidence.round(ROUND_PRECISION).alias("reverse_confidence"),
        lift.round(ROUND_PRECISION).alias("lift"),
        leverage.round(ROUND_PRECISION).alias("leverage"),
        confidence_laplace.round(ROUND_PRECISION).alias("confidence_laplace"),
        confidence_lower.round(ROUND_PRECISION).alias("confidence_lower"),
        confidence_upper.round(ROUND_PRECISION).alias("confidence_upper"),
        confidence_lower_laplace.round(ROUND_PRECISION).alias("confidence_lower_laplace"),
        confidence_upper_laplace.round(ROUND_PRECISION).alias("confidence_upper_laplace"),
        zhangs_metric.round(ROUND_PRECISION).alias("zhangs_metric"),
        importance.round(ROUND_PRECISION).alias("importance"),
        importance_laplace.round(ROUND_PRECISION).alias("importance_laplace"),
        added_value.round(ROUND_PRECISION).alias("added_value"),
        improvement.round(ROUND_PRECISION).alias("improvement"),
        cosine.round(ROUND_PRECISION).alias("cosine"),
        conviction.round(ROUND_PRECISION).alias("conviction"),
        jaccard.round(ROUND_PRECISION).alias("jaccard"),
        kulczynski.round(ROUND_PRECISION).alias("kulczynski"),
        klosgen.round(ROUND_PRECISION).alias("klosgen"),
        relative_difference.round(ROUND_PRECISION).alias("relative_difference"),
        rule_power_factor.round(ROUND_PRECISION).alias("rule_power_factor"),
        weight_of_evidence.round(ROUND_PRECISION).alias("weight_of_evidence"),
        casual_confidence.round(ROUND_PRECISION).alias("casual_confidence"),
        casual_support.round(ROUND_PRECISION).alias("casual_support"),
        confirmed_confidence.round(ROUND_PRECISION).alias("confirmed_confidence"),
        counter_example_rate.round(ROUND_PRECISION).alias("counter_example_rate"),
        implication_index.round(ROUND_PRECISION).alias("implication_index"),
        lambda_measure.round(ROUND_PRECISION).alias("lambda"),
        least_contradiction.round(ROUND_PRECISION).alias("least_contradiction"),
        lerman_similarity.round(ROUND_PRECISION).alias("lerman_similarity"),
        lift_increase.round(ROUND_PRECISION).alias("lift_increase"),
        mutual_information.round(ROUND_PRECISION).alias("mutual_information"),
        rld.round(ROUND_PRECISION).alias("relative_linkage_disequilibrium"),
        relative_risk.round(ROUND_PRECISION).alias("relative_risk"),
        standardized_lift.round(ROUND_PRECISION).alias("standardized_lift"),
        sebag_schoenauer.round(ROUND_PRECISION).alias("sebag_schoenauer"),
        varying_rates_liaison.round(ROUND_PRECISION).alias("varying_rates_liaison"),
        support_vrl.round(ROUND_PRECISION).alias("support_vrl"),
        collective_strength.round(ROUND_PRECISION).alias("collective_strength"),
        confidence_boost.round(ROUND_PRECISION).alias("confidence_boost"),
        odds_ratio.round(ROUND_PRECISION).alias("odds_ratio"),
        odds_ratio_lower.round(ROUND_PRECISION).alias("odds_ratio_lower"),
        odds_ratio_upper.round(ROUND_PRECISION).alias("odds_ratio_upper"),
        certainty_factor.round(ROUND_PRECISION).alias("certainty_factor"),
        imbalance_ratio.round(ROUND_PRECISION).alias("imbalance_ratio"),
        gini_index.round(ROUND_PRECISION).alias("gini_index"),
        hyper_confidence.round(ROUND_PRECISION).alias("hyper_confidence"),
        hyper_lift.round(ROUND_PRECISION).alias("hyper_lift"),
        chi_sq.round(ROUND_PRECISION).alias("chi_squared"),
        p_value_approx.round(ROUND_PRECISION).alias("p_value_approximation"),
        local_chi_sq.round(ROUND_PRECISION).alias("local_chi_squared"),
        phi_coefficient.round(ROUND_PRECISION).alias("phi_coefficient"),
        yules_q.round(ROUND_PRECISION).alias("yules_q"),
        yules_y.round(ROUND_PRECISION).alias("yules_y"),
        kappa.round(ROUND_PRECISION).alias("kappa"),
        j_measure.round(ROUND_PRECISION).alias("j_measure"),
        difference_of_confidence.round(ROUND_PRECISION).alias("difference_of_confidence"),
        hamming.round(ROUND_PRECISION).alias("hamming"),
        rogers_tanimoto.round(ROUND_PRECISION).alias("rogers_tanimoto"),
        sokal_michener.round(ROUND_PRECISION).alias("sokal_michener"),
        sokal_sneath.round(ROUND_PRECISION).alias("sokal_sneath"),
        interestingness.round(ROUND_PRECISION).alias("interestingness"),
        comprehensibility.round(ROUND_PRECISION).alias("comprehensibility"),
        fisher_conf.round(ROUND_PRECISION).alias("fisher_transformation_confidence"),
        fisher_rev_conf.round(ROUND_PRECISION).alias("fisher_transformation_reverse_confidence"),
    )

    return result.collect()
