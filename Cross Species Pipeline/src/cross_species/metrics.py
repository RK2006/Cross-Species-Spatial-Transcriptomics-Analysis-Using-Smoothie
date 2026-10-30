"""Scalar objective metrics derived from one clustering evaluation.

The expensive calculations belong in :func:`evaluate_joint_modules`.  The
functions here consume its gene- and module-level tables and return one scalar
each.  This keeps every objective coordinate independently documented without
recomputing the underlying evaluation for every metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


Evaluation = Mapping[str, object]
MetricFunction = Callable[..., float | int]


def _table(evaluation: Evaluation, name: str) -> pd.DataFrame:
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping returned by evaluate_joint_modules")
    value = evaluation.get(name)
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"evaluation must contain a {name!r} DataFrame")
    return value


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"required evaluation column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric_column(frame, column)
    return values[np.isfinite(values)]


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else np.nan


def _meaningful_modules(
    evaluation: Evaluation, min_module_size: int
) -> pd.DataFrame:
    if not isinstance(min_module_size, (int, np.integer)) or min_module_size < 1:
        raise ValueError("min_module_size must be a positive integer")
    modules = _table(evaluation, "modules")
    sizes = _numeric_column(modules, "n_genes")
    return modules.loc[np.isfinite(sizes) & (sizes >= min_module_size)]


def _cohesion_column(modules: pd.DataFrame, first: str, second: str) -> str:
    columns = {str(column).casefold(): str(column) for column in modules.columns}
    candidates = [
        f"cohesion_{first}_{second}".casefold(),
        f"cohesion_{second}_{first}".casefold(),
    ]
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    raise ValueError(
        f"no cohesion column was found for {first!r} and {second!r}; "
        f"available cohesion columns are "
        f"{[column for column in modules if str(column).startswith('cohesion_')]}"
    )


def median_gene_margin(evaluation: Evaluation) -> float:
    """Median gene margin over genes with a defined margin."""
    return _median(_finite_column(_table(evaluation, "genes"), "gene_margin"))


def p10_gene_margin(evaluation: Evaluation) -> float:
    """10th percentile of gene margin over genes with a defined margin."""
    values = _finite_column(_table(evaluation, "genes"), "gene_margin")
    return float(np.quantile(values, 0.10)) if len(values) else np.nan


def fraction_negative_margin(evaluation: Evaluation) -> float:
    """Fraction of genes with defined margins whose margin is below zero."""
    values = _finite_column(_table(evaluation, "genes"), "gene_margin")
    return float(np.mean(values < 0)) if len(values) else np.nan


def median_cross_strength_fraction(evaluation: Evaluation) -> float:
    """Median cross-strength/total-strength ratio over non-isolated genes."""
    values = _finite_column(
        _table(evaluation, "genes"), "cross_strength_fraction"
    )
    return _median(values)


def cross_edge_coverage(evaluation: Evaluation) -> float:
    """Fraction of all genes with at least one cross-species graph edge."""
    genes = _table(evaluation, "genes")
    values = _numeric_column(genes, "cross_degree")
    if not len(values):
        return np.nan
    if not np.isfinite(values).all():
        raise ValueError("cross_degree contains missing or non-numeric values")
    return float(np.mean(values > 0))


def reciprocal_gene_coverage(evaluation: Evaluation) -> float:
    """Fraction of all genes with at least one reciprocal cross-neighbor."""
    genes = _table(evaluation, "genes")
    values = _numeric_column(genes, "reciprocal_cross_degree")
    if not len(values):
        return np.nan
    if not np.isfinite(values).all():
        raise ValueError(
            "reciprocal_cross_degree contains missing or non-numeric values"
        )
    return float(np.mean(values > 0))


def median_mouse_cohesion(
    evaluation: Evaluation, *, min_module_size: int = 10
) -> float:
    """Median mouse-mouse cohesion across meaningful modules where defined."""
    modules = _meaningful_modules(evaluation, min_module_size)
    column = _cohesion_column(modules, "mouse", "mouse")
    return _median(_finite_column(modules, column))


def median_human_cohesion(
    evaluation: Evaluation, *, min_module_size: int = 10
) -> float:
    """Median human-human cohesion across meaningful modules where defined."""
    modules = _meaningful_modules(evaluation, min_module_size)
    column = _cohesion_column(modules, "human", "human")
    return _median(_finite_column(modules, column))


def median_cross_species_cohesion(
    evaluation: Evaluation, *, min_module_size: int = 10
) -> float:
    """Median mouse-human cohesion across meaningful mixed modules."""
    modules = _meaningful_modules(evaluation, min_module_size)
    column = _cohesion_column(modules, "mouse", "human")
    return _median(_finite_column(modules, column))


def median_conserved_diversity(
    evaluation: Evaluation, *, min_module_size: int = 10
) -> float:
    """Median minimum(mouse diversity, human diversity) across meaningful modules.

    Modules need a defined spatial diversity value in both species.  The
    evaluator creates ``conserved_diversity`` as their elementwise minimum.
    """
    modules = _meaningful_modules(evaluation, min_module_size)
    return _median(_finite_column(modules, "conserved_diversity"))


def fraction_isolated_genes(evaluation: Evaluation) -> float:
    """Fraction of all genes with total graph degree equal to zero."""
    genes = _table(evaluation, "genes")
    values = _numeric_column(genes, "degree")
    if not len(values):
        return np.nan
    if not np.isfinite(values).all():
        raise ValueError("degree contains missing or non-numeric values")
    return float(np.mean(values == 0))


def fraction_genes_in_largest_module(evaluation: Evaluation) -> float:
    """Fraction of all evaluated genes assigned to the largest module."""
    genes = _table(evaluation, "genes")
    modules = _table(evaluation, "modules")
    if len(genes) == 0:
        return np.nan
    sizes = _finite_column(modules, "n_genes")
    return float(sizes.max() / len(genes)) if len(sizes) else np.nan


def number_meaningful_modules(
    evaluation: Evaluation, *, min_module_size: int = 10
) -> int:
    """Number of modules containing at least ``min_module_size`` genes."""
    return int(len(_meaningful_modules(evaluation, min_module_size)))


@dataclass(frozen=True)
class MetricDefinition:
    """Metadata and calculation rule for one coordinate of the metric vector."""

    name: str
    category: str
    direction: str
    level: str
    description: str
    function: MetricFunction
    uses_meaningful_modules: bool = False

    def calculate(
        self, evaluation: Evaluation, *, min_module_size: int = 10
    ) -> float | int:
        if self.uses_meaningful_modules:
            return self.function(evaluation, min_module_size=min_module_size)
        return self.function(evaluation)


METRIC_DEFINITIONS = (
    MetricDefinition(
        "median_gene_margin", "separation", "maximize", "gene",
        "Median margin among genes with a defined margin.",
        median_gene_margin,
    ),
    MetricDefinition(
        "p10_gene_margin", "separation", "maximize", "gene",
        "10th percentile of defined gene margins.",
        p10_gene_margin,
    ),
    MetricDefinition(
        "fraction_negative_margin", "separation", "minimize", "gene",
        "Fraction of genes with defined margins below zero.",
        fraction_negative_margin,
    ),
    MetricDefinition(
        "median_cross_strength_fraction", "integration", "target", "gene",
        "Median fraction of graph strength contributed by cross-species edges.",
        median_cross_strength_fraction,
    ),
    MetricDefinition(
        "cross_edge_coverage", "integration", "maximize", "gene",
        "Fraction of all genes with at least one cross-species graph edge.",
        cross_edge_coverage,
    ),
    MetricDefinition(
        "reciprocal_gene_coverage", "integration", "maximize", "gene",
        "Fraction of all genes with reciprocal cross-neighbor support.",
        reciprocal_gene_coverage,
    ),
    MetricDefinition(
        "median_mouse_cohesion", "module_quality", "maximize", "module",
        "Median mouse-mouse cohesion across meaningful modules.",
        median_mouse_cohesion, True,
    ),
    MetricDefinition(
        "median_human_cohesion", "module_quality", "maximize", "module",
        "Median human-human cohesion across meaningful modules.",
        median_human_cohesion, True,
    ),
    MetricDefinition(
        "median_cross_species_cohesion", "module_quality", "maximize", "module",
        "Median mouse-human cohesion across meaningful mixed modules.",
        median_cross_species_cohesion, True,
    ),
    MetricDefinition(
        "median_conserved_diversity", "module_quality", "maximize", "module",
        "Median minimum mouse/human spatial diversity across meaningful modules.",
        median_conserved_diversity, True,
    ),
    MetricDefinition(
        "fraction_isolated_genes", "module_structure", "minimize", "gene",
        "Fraction of all genes with graph degree zero.",
        fraction_isolated_genes,
    ),
    MetricDefinition(
        "fraction_genes_in_largest_module", "module_structure", "target", "gene",
        "Fraction of all genes assigned to the largest module.",
        fraction_genes_in_largest_module,
    ),
    MetricDefinition(
        "number_meaningful_modules", "module_structure", "target", "module",
        "Number of modules at or above the meaningful-module size threshold.",
        number_meaningful_modules, True,
    ),
)

METRIC_REGISTRY = {definition.name: definition for definition in METRIC_DEFINITIONS}


def metric_vector(
    evaluation: Evaluation,
    *,
    min_module_size: int = 10,
    metrics: Sequence[str] | None = None,
) -> pd.Series:
    """Calculate a named metric vector from one clustering evaluation.

    Parameters
    ----------
    evaluation
        Dictionary returned by ``evaluate_joint_modules``.
    min_module_size
        Threshold used for all module-level medians and for the meaningful
        module count. Gene-level metrics always retain all applicable genes.
    metrics
        Optional ordered metric-name subset. The full 13-metric registry is
        used by default.
    """
    names = list(METRIC_REGISTRY) if metrics is None else list(metrics)
    unknown = [name for name in names if name not in METRIC_REGISTRY]
    if unknown:
        raise KeyError(f"unknown metrics: {unknown}")
    if len(names) != len(set(names)):
        raise ValueError("metrics contains duplicate names")
    values = {
        name: METRIC_REGISTRY[name].calculate(
            evaluation, min_module_size=min_module_size
        )
        for name in names
    }
    result = pd.Series(values, name="value")
    result.attrs["min_module_size"] = int(min_module_size)
    return result


def metric_metadata() -> pd.DataFrame:
    """Return the registry as a human-readable table."""
    return pd.DataFrame([
        {
            "metric": definition.name,
            "category": definition.category,
            "direction": definition.direction,
            "level": definition.level,
            "uses_meaningful_modules": definition.uses_meaningful_modules,
            "description": definition.description,
        }
        for definition in METRIC_DEFINITIONS
    ])


__all__ = [
    "METRIC_DEFINITIONS",
    "METRIC_REGISTRY",
    "MetricDefinition",
    "cross_edge_coverage",
    "fraction_genes_in_largest_module",
    "fraction_isolated_genes",
    "fraction_negative_margin",
    "median_conserved_diversity",
    "median_cross_species_cohesion",
    "median_cross_strength_fraction",
    "median_gene_margin",
    "median_human_cohesion",
    "median_mouse_cohesion",
    "metric_metadata",
    "metric_vector",
    "number_meaningful_modules",
    "p10_gene_margin",
    "reciprocal_gene_coverage",
]
