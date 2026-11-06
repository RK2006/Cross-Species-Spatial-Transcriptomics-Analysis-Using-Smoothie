"""Transparent first-pass desirability scoring for clustering metrics.

Fit a :class:`ScoreSpecification` once on an initial calibration search, freeze
it, and use :func:`score_objective` for every candidate that follows.  The
metric, category, and final scores are all retained so the objective remains
auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


# These indicators give each scientific idea a small number of representatives.
# The remaining coordinates in the 65-value vector remain available for plots
# and later revisions without automatically receiving equal weight.
FIRST_PASS_METRICS = {
    "separation": (
        ("worst_species_p10_gene_margin", "maximize"),
        ("median_gene_margin", "maximize"),
        ("fraction_negative_margin", "minimize"),
        ("absolute_species_median_margin_difference", "minimize"),
    ),
    "integration": (
        ("median_cross_strength_fraction", "maximize"),
        ("cross_edge_coverage", "maximize"),
        ("reciprocal_gene_coverage", "maximize"),
        ("fraction_genes_in_mixed_meaningful_modules", "maximize"),
        ("p25_module_species_balance", "maximize"),
    ),
    "module_quality": (
        ("p10_cross_species_cohesion", "maximize"),
        ("median_cross_species_cohesion", "maximize"),
        ("p10_conserved_diversity", "maximize"),
        ("median_conserved_diversity", "maximize"),
        ("fraction_meaningful_modules_with_cross_species_cohesion", "maximize"),
        ("fraction_meaningful_modules_with_both_diversities", "maximize"),
    ),
    "module_structure": (
        ("fraction_isolated_genes", "minimize"),
        ("fraction_genes_in_largest_module", "target"),
        ("number_meaningful_modules", "target"),
    ),
}


@dataclass(frozen=True)
class DesirabilityRule:
    """Frozen transformation from one raw metric to the interval [0, 1]."""

    metric: str
    category: str
    family: str
    direction: str
    outer_low: float
    outer_high: float
    target_low: float | None = None
    target_high: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize", "target"}:
            raise ValueError(f"unsupported direction: {self.direction}")
        if not np.isfinite([self.outer_low, self.outer_high]).all():
            raise ValueError(f"non-finite bounds for {self.metric}")
        if self.outer_high <= self.outer_low:
            raise ValueError(f"outer bounds must increase for {self.metric}")
        if self.direction == "target":
            if self.target_low is None or self.target_high is None:
                raise ValueError(f"target bounds are required for {self.metric}")
            if not (
                self.outer_low <= self.target_low
                <= self.target_high <= self.outer_high
            ):
                raise ValueError(f"invalid target bounds for {self.metric}")

    def transform(self, value: float) -> float:
        """Convert one finite raw value to desirability in [0, 1]."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{self.metric} is missing or non-finite")
        width = self.outer_high - self.outer_low
        if self.direction == "maximize":
            return float(np.clip((value - self.outer_low) / width, 0.0, 1.0))
        if self.direction == "minimize":
            return float(np.clip((self.outer_high - value) / width, 0.0, 1.0))

        assert self.target_low is not None and self.target_high is not None
        if self.target_low <= value <= self.target_high:
            return 1.0
        if value < self.target_low:
            width = self.target_low - self.outer_low
            if width == 0:
                return float(value >= self.target_low)
            return float(np.clip((value - self.outer_low) / width, 0.0, 1.0))
        width = self.outer_high - self.target_high
        if width == 0:
            return float(value <= self.target_high)
        return float(np.clip((self.outer_high - value) / width, 0.0, 1.0))


@dataclass(frozen=True)
class ScoreSpecification:
    """Frozen rules, category weights, and optional feasibility constraints."""

    rules: tuple[DesirabilityRule, ...]
    category_weights: Mapping[str, float]
    constraints: Mapping[str, tuple[float | None, float | None]]
    calibration_quantiles: tuple[float, float, float, float] | None

    def rules_frame(self) -> pd.DataFrame:
        """Return the frozen transformation rules as a table."""
        return pd.DataFrame([
            {
                "metric": rule.metric,
                "category": rule.category,
                "family": rule.family,
                "direction": rule.direction,
                "outer_low": rule.outer_low,
                "target_low": rule.target_low,
                "target_high": rule.target_high,
                "outer_high": rule.outer_high,
            }
            for rule in self.rules
        ])


@dataclass(frozen=True)
class ObjectiveScore:
    """Final score together with all values needed to explain it."""

    final_score: float
    category_scores: pd.Series
    family_scores: pd.Series
    desirabilities: pd.Series
    feasible: bool
    violations: tuple[str, ...]


def educated_full_score_spec(
    *,
    category_weights: Mapping[str, float] | None = None,
    constraints: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> ScoreSpecification:
    """Return provisional scientific targets for all 65 objective indicators.

    The targets are deliberately broad first estimates based on the natural
    metric ranges, the interpretation of each statistic, and the behavior of
    the existing run-049 clustering. They are starting hypotheses, not fitted
    biological truths. Review the desirability and family-score distributions
    after the first Sobol experiment and revise targets before final selection.
    """
    rules: list[DesirabilityRule] = []

    def add(
        metric: str,
        category: str,
        family: str,
        direction: str,
        outer_low: float,
        outer_high: float,
        target_low: float | None = None,
        target_high: float | None = None,
    ) -> None:
        rules.append(DesirabilityRule(
            metric=metric,
            category=category,
            family=family,
            direction=direction,
            outer_low=outer_low,
            outer_high=outer_high,
            target_low=target_low,
            target_high=target_high,
        ))

    labels = ("p10", "p25", "median", "p75", "p90")

    # Separation: negative/low margins are poor; upper-tail targets approach 1.
    for label, low, target in zip(
        labels,
        (-0.10, 0.10, 0.30, 0.50, 0.65),
        (0.40, 0.60, 0.82, 0.93, 0.98),
    ):
        add(f"{label}_gene_margin", "separation", "gene_margin", "maximize", low, target)
    add("fraction_negative_margin", "separation", "negative_margin", "minimize", 0.03, 0.15)
    add("worst_species_p10_gene_margin", "separation", "species_margin_floor", "maximize", -0.10, 0.35)
    add("absolute_species_median_margin_difference", "separation", "species_margin_parity", "minimize", 0.03, 0.20)

    # Integration: cross strength has a broad target band because either almost
    # no cross signal or overwhelming cross signal can be undesirable.
    cross_strength_ranges = (
        (0.00, 0.05, 0.25, 0.55),
        (0.02, 0.15, 0.40, 0.70),
        (0.10, 0.30, 0.55, 0.80),
        (0.20, 0.40, 0.70, 0.90),
        (0.30, 0.55, 0.85, 0.98),
    )
    for label, (outer_low, target_low, target_high, outer_high) in zip(labels, cross_strength_ranges):
        add(
            f"{label}_cross_strength_fraction", "integration", "cross_strength",
            "target", outer_low, outer_high, target_low, target_high,
        )
    for label, low, target in zip(
        labels,
        (0.00, 0.00, 0.00, 0.02, 0.08),
        (0.05, 0.10, 0.20, 0.35, 0.55),
    ):
        add(
            f"{label}_reciprocal_cross_fraction", "integration",
            "reciprocal_support", "maximize", low, target,
        )
    add("cross_edge_coverage", "integration", "cross_coverage", "maximize", 0.50, 0.90)
    add("reciprocal_gene_coverage", "integration", "cross_coverage", "maximize", 0.20, 0.70)
    add("fraction_genes_in_mixed_meaningful_modules", "integration", "mixed_module_coverage", "maximize", 0.50, 0.90)
    for label, low, target in zip(
        labels,
        (0.00, 0.00, 0.10, 0.30, 0.50),
        (0.15, 0.25, 0.50, 0.70, 0.85),
    ):
        add(f"{label}_module_species_balance", "integration", "module_species_balance", "maximize", low, target)

    # Quality: within-species cohesion is expected to be higher than cross-
    # species cohesion, so the latter receives slightly lower targets.
    within_lows = (0.40, 0.50, 0.60, 0.70, 0.75)
    within_targets = (0.70, 0.75, 0.82, 0.88, 0.92)
    cross_lows = (0.20, 0.30, 0.40, 0.50, 0.60)
    cross_targets = (0.50, 0.60, 0.70, 0.78, 0.85)
    for species in ("mouse", "human"):
        for label, low, target in zip(labels, within_lows, within_targets):
            add(f"{label}_{species}_cohesion", "quality", f"{species}_cohesion", "maximize", low, target)
    for label, low, target in zip(labels, cross_lows, cross_targets):
        add(f"{label}_cross_species_cohesion", "quality", "cross_species_cohesion", "maximize", low, target)

    diversity_lows = (0.00, 0.03, 0.08, 0.15, 0.25)
    diversity_targets = (0.12, 0.22, 0.38, 0.58, 0.75)
    conserved_lows = (0.00, 0.02, 0.05, 0.10, 0.18)
    conserved_targets = (0.08, 0.16, 0.28, 0.45, 0.60)
    for species in ("mouse", "human"):
        for label, low, target in zip(labels, diversity_lows, diversity_targets):
            add(f"{label}_{species}_diversity", "quality", f"{species}_diversity", "maximize", low, target)
    for label, low, target in zip(labels, conserved_lows, conserved_targets):
        add(f"{label}_conserved_diversity", "quality", "conserved_diversity", "maximize", low, target)

    add("fraction_meaningful_modules_with_mouse_cohesion", "quality", "cohesion_coverage", "maximize", 0.40, 0.90)
    add("fraction_meaningful_modules_with_human_cohesion", "quality", "cohesion_coverage", "maximize", 0.40, 0.90)
    add("fraction_meaningful_modules_with_cross_species_cohesion", "quality", "cohesion_coverage", "maximize", 0.30, 0.80)
    add("fraction_meaningful_modules_with_both_diversities", "quality", "diversity_coverage", "maximize", 0.30, 0.80)
    add("p10_cross_vs_within_cohesion_contrast", "quality", "cohesion_contrast", "target", -0.60, 0.25, -0.20, 0.05)
    add("median_cross_vs_within_cohesion_contrast", "quality", "cohesion_contrast", "target", -0.45, 0.20, -0.12, 0.05)

    # Structure targets are broad for a 15k-gene network and should be revisited
    # if the number of genes or the meaningful-module threshold changes.
    add("fraction_isolated_genes", "structure", "isolation", "minimize", 0.02, 0.12)
    add("fraction_genes_in_largest_module", "structure", "largest_module", "target", 0.02, 0.40, 0.07, 0.20)
    add("number_meaningful_modules", "structure", "module_count", "target", 5.0, 80.0, 15.0, 45.0)

    from .metrics import METRIC_REGISTRY

    expected = set(METRIC_REGISTRY)
    actual = {rule.metric for rule in rules}
    if actual != expected or len(rules) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"full score rules do not match metric registry; missing={missing}, extra={extra}")

    categories = ("separation", "integration", "quality", "structure")
    weights = (
        {category: 1.0 for category in categories}
        if category_weights is None else dict(category_weights)
    )
    if set(weights) != set(categories):
        raise ValueError(f"category_weights must contain exactly {list(categories)}")
    if any(not np.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("all category weights must be positive and finite")

    default_constraints = {
        "fraction_isolated_genes": (None, 0.20),
        "cross_edge_coverage": (0.30, None),
        "fraction_genes_in_largest_module": (None, 0.50),
        "number_meaningful_modules": (5.0, None),
    }
    return ScoreSpecification(
        rules=tuple(rules),
        category_weights=weights,
        constraints=default_constraints if constraints is None else dict(constraints),
        calibration_quantiles=None,
    )


def _as_metric_frame(calibration: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(calibration, pd.DataFrame) or calibration.empty:
        raise ValueError("calibration must be a non-empty DataFrame")
    return calibration.apply(pd.to_numeric, errors="coerce")


def fit_first_pass_score_spec(
    calibration: pd.DataFrame,
    *,
    metric_groups: Mapping[str, Sequence[tuple[str, str]]] = FIRST_PASS_METRICS,
    calibration_quantiles: tuple[float, float, float, float] = (0.10, 0.25, 0.75, 0.90),
    target_ranges: Mapping[str, tuple[float, float]] | None = None,
    category_weights: Mapping[str, float] | None = None,
    constraints: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> ScoreSpecification:
    """Fit and freeze a first-pass scoring specification.

    ``calibration`` has one row per clustering and one column per raw metric.
    Maximize/minimize metrics receive robust outer bounds from the 10th and
    90th percentiles by default. Target metrics use the 25th--75th percentile
    interval as a provisional plateau unless an explicit range is supplied.
    These empirical target ranges are suitable for an initial pipeline test;
    replace them with scientifically chosen ranges before a definitive search.
    """
    frame = _as_metric_frame(calibration)
    q_outer_low, q_target_low, q_target_high, q_outer_high = calibration_quantiles
    if not (0 <= q_outer_low < q_target_low <= q_target_high < q_outer_high <= 1):
        raise ValueError("calibration_quantiles must be ordered within [0, 1]")
    target_ranges = dict(target_ranges or {})

    rules: list[DesirabilityRule] = []
    for category, specifications in metric_groups.items():
        for metric, direction in specifications:
            if metric not in frame:
                raise ValueError(f"calibration is missing required metric: {metric}")
            values = frame[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 2:
                raise ValueError(f"at least two finite calibration values are required for {metric}")
            outer_low, outer_high = np.quantile(values, [q_outer_low, q_outer_high])
            if outer_high <= outer_low:
                # A constant metric carries no ranking information. A tiny
                # symmetric interval assigns every observed value the midpoint.
                scale = max(abs(float(outer_low)), 1.0)
                outer_low -= np.finfo(float).eps * scale
                outer_high += np.finfo(float).eps * scale

            target_low = target_high = None
            if direction == "target":
                if metric in target_ranges:
                    target_low, target_high = map(float, target_ranges[metric])
                    outer_low = min(float(outer_low), target_low)
                    outer_high = max(float(outer_high), target_high)
                else:
                    target_low, target_high = np.quantile(
                        values, [q_target_low, q_target_high]
                    )
            rules.append(DesirabilityRule(
                metric=metric,
                category=str(category),
                family=metric,
                direction=direction,
                outer_low=float(outer_low),
                outer_high=float(outer_high),
                target_low=None if target_low is None else float(target_low),
                target_high=None if target_high is None else float(target_high),
            ))

    categories = list(metric_groups)
    weights = (
        {category: 1.0 for category in categories}
        if category_weights is None else dict(category_weights)
    )
    if set(weights) != set(categories):
        raise ValueError("category_weights must contain exactly the score categories")
    if any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("all category weights must be positive and finite")

    return ScoreSpecification(
        rules=tuple(rules),
        category_weights=weights,
        constraints=dict(constraints or {}),
        calibration_quantiles=tuple(map(float, calibration_quantiles)),
    )


def _metric_series(metric_values: Mapping[str, float] | pd.Series) -> pd.Series:
    values = metric_values if isinstance(metric_values, pd.Series) else pd.Series(metric_values)
    return pd.to_numeric(values, errors="coerce")


def score_objective(
    metric_values: Mapping[str, float] | pd.Series,
    specification: ScoreSpecification,
) -> ObjectiveScore:
    """Produce one final score from a named metric vector and frozen rules.

    Desirabilities are averaged within metric families, then families are
    averaged within categories. Category scores are combined with a weighted
    geometric mean. A failed feasibility constraint sets the final score to
    zero but retains all diagnostic values.
    """
    values = _metric_series(metric_values)
    desirabilities = {}
    family_members: dict[tuple[str, str], list[float]] = {}
    for rule in specification.rules:
        if rule.metric not in values:
            raise ValueError(f"metric vector is missing required metric: {rule.metric}")
        desirability = rule.transform(values[rule.metric])
        desirabilities[rule.metric] = desirability
        family_members.setdefault((rule.category, rule.family), []).append(
            desirability
        )

    family_scores = pd.Series(
        {
            (category, family): float(np.mean(members))
            for (category, family), members in family_members.items()
        },
        name="family_score",
    )
    family_scores.index = pd.MultiIndex.from_tuples(
        family_scores.index, names=["category", "family"]
    )
    category_scores = family_scores.groupby(level="category", sort=False).mean()
    category_scores.name = "category_score"

    violations = []
    for metric, (minimum, maximum) in specification.constraints.items():
        if metric not in values or not np.isfinite(values[metric]):
            violations.append(f"{metric}: missing")
            continue
        value = float(values[metric])
        if minimum is not None and value < minimum:
            violations.append(f"{metric}: {value:g} < {minimum:g}")
        if maximum is not None and value > maximum:
            violations.append(f"{metric}: {value:g} > {maximum:g}")

    weights = pd.Series(specification.category_weights, dtype=float)
    ordered_scores = category_scores.reindex(weights.index)
    if ordered_scores.isna().any():
        missing = ordered_scores.index[ordered_scores.isna()].tolist()
        raise ValueError(f"category scores are missing: {missing}")
    if (ordered_scores == 0).any():
        combined = 0.0
    else:
        combined = float(np.exp(np.average(np.log(ordered_scores), weights=weights)))
    feasible = not violations
    return ObjectiveScore(
        final_score=combined if feasible else 0.0,
        category_scores=category_scores,
        family_scores=family_scores,
        desirabilities=pd.Series(desirabilities, name="desirability"),
        feasible=feasible,
        violations=tuple(violations),
    )


def score_objective_table(
    metric_table: pd.DataFrame,
    specification: ScoreSpecification,
) -> pd.DataFrame:
    """Score every row of a metric table and return compact run-level results."""
    rows = []
    for index, metric_values in metric_table.iterrows():
        result = score_objective(metric_values, specification)
        row = {
            "run": index,
            "final_score": result.final_score,
            "feasible": result.feasible,
            "violations": "; ".join(result.violations),
        }
        row.update({f"score_{name}": value for name, value in result.category_scores.items()})
        row.update({
            f"family_{category}__{family}": value
            for (category, family), value in result.family_scores.items()
        })
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


__all__ = [
    "DesirabilityRule",
    "educated_full_score_spec",
    "FIRST_PASS_METRICS",
    "ObjectiveScore",
    "ScoreSpecification",
    "fit_first_pass_score_spec",
    "score_objective",
    "score_objective_table",
]
