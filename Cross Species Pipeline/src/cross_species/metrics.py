"""Named scalar coordinates for the clustering objective vector.

The expensive work is performed once by ``evaluate_joint_modules``. Every
function here reads its gene- or module-level tables and returns one scalar.
Quantiles use NumPy's default linear interpolation and ignore non-finite values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

Evaluation = Mapping[str, object]
MetricFunction = Callable[..., float | int]
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_LABELS = ("p10", "p25", "median", "p75", "p90")


def _table(evaluation: Evaluation, name: str) -> pd.DataFrame:
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping returned by evaluate_joint_modules")
    value = evaluation.get(name)
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"evaluation must contain a {name!r} DataFrame")
    return value


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"required evaluation column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric(frame, column)
    return values[np.isfinite(values)]


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else np.nan


def _meaningful(evaluation: Evaluation, min_module_size: int) -> pd.DataFrame:
    if not isinstance(min_module_size, (int, np.integer)) or min_module_size < 1:
        raise ValueError("min_module_size must be a positive integer")
    modules = _table(evaluation, "modules")
    sizes = _numeric(modules, "n_genes")
    return modules.loc[np.isfinite(sizes) & (sizes >= min_module_size)].copy()


def _find_column(frame: pd.DataFrame, *candidates: str) -> str:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise ValueError(f"none of the required columns were found: {list(candidates)}")


def _cohesion_columns(modules: pd.DataFrame) -> tuple[str, str, str]:
    return (
        _find_column(modules, "cohesion_mouse_mouse"),
        _find_column(modules, "cohesion_human_human"),
        _find_column(modules, "cohesion_mouse_human", "cohesion_human_mouse"),
    )


def _diversity_columns(modules: pd.DataFrame) -> tuple[str, str, str]:
    return (
        _find_column(modules, "diversity_mouse"),
        _find_column(modules, "diversity_human"),
        _find_column(modules, "conserved_diversity"),
    )


def _shared_cohesion_modules(evaluation: Evaluation, min_module_size: int):
    modules = _meaningful(evaluation, min_module_size)
    columns = _cohesion_columns(modules)
    valid = (
        (_numeric(modules, _find_column(modules, "n_mouse")) >= 2)
        & (_numeric(modules, _find_column(modules, "n_human")) >= 2)
    )
    for column in columns:
        valid &= np.isfinite(_numeric(modules, column))
    return modules.loc[valid].copy(), columns


def _shared_diversity_modules(evaluation: Evaluation, min_module_size: int):
    modules = _meaningful(evaluation, min_module_size)
    columns = _diversity_columns(modules)
    valid = np.ones(len(modules), dtype=bool)
    for column in columns:
        valid &= np.isfinite(_numeric(modules, column))
    return modules.loc[valid].copy(), columns


def _make_gene_quantile(name: str, column: str, q: float, description: str):
    def function(evaluation: Evaluation) -> float:
        return _quantile(_finite(_table(evaluation, "genes"), column), q)
    function.__name__ = name
    function.__doc__ = description
    return function


def _make_module_quantile(name: str, family: str, member: int, q: float, description: str):
    def function(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
        if family == "cohesion":
            modules, columns = _shared_cohesion_modules(evaluation, min_module_size)
        else:
            modules, columns = _shared_diversity_modules(evaluation, min_module_size)
        return _quantile(_finite(modules, columns[member]), q)
    function.__name__ = name
    function.__doc__ = description
    return function


for _label, _q in zip(QUANTILE_LABELS, QUANTILES):
    for _column, _stem, _description in (
        ("gene_margin", "gene_margin", "gene margin"),
        ("cross_strength_fraction", "cross_strength_fraction", "cross-strength fraction"),
        ("reciprocal_cross_fraction", "reciprocal_cross_fraction", "reciprocal cross-neighbor fraction"),
    ):
        _name = f"{_label}_{_stem}"
        globals()[_name] = _make_gene_quantile(
            _name, _column, _q, f"{_label} quantile of defined {_description} values."
        )
    for _family, _members in (
        ("cohesion", ("mouse", "human", "cross_species")),
        ("diversity", ("mouse", "human", "conserved")),
    ):
        for _member, _stem in enumerate(_members):
            _name = f"{_label}_{_stem}_{_family}"
            globals()[_name] = _make_module_quantile(
                _name, _family, _member, _q,
                f"{_label} quantile of {_stem.replace('_', ' ')} {_family} on the shared eligible module set.",
            )


def fraction_negative_margin(evaluation: Evaluation) -> float:
    """Fraction of genes with a defined margin whose margin is negative."""
    values = _finite(_table(evaluation, "genes"), "gene_margin")
    return float(np.mean(values < 0)) if values.size else np.nan


def cross_edge_coverage(evaluation: Evaluation) -> float:
    """Fraction of all genes with at least one cross-species edge."""
    values = _numeric(_table(evaluation, "genes"), "cross_degree")
    if not values.size:
        return np.nan
    if not np.isfinite(values).all():
        raise ValueError("cross_degree contains missing or non-numeric values")
    return float(np.mean(values > 0))


def reciprocal_gene_coverage(evaluation: Evaluation) -> float:
    """Fraction of all genes with at least one reciprocal cross-neighbor."""
    values = _numeric(_table(evaluation, "genes"), "reciprocal_cross_degree")
    if not values.size:
        return np.nan
    if not np.isfinite(values).all():
        raise ValueError("reciprocal_cross_degree contains missing values")
    return float(np.mean(values > 0))


def fraction_isolated_genes(evaluation: Evaluation) -> float:
    """Fraction of all genes with total graph degree zero."""
    values = _numeric(_table(evaluation, "genes"), "degree")
    if not values.size:
        return np.nan
    if not np.isfinite(values).all():
        raise ValueError("degree contains missing or non-numeric values")
    return float(np.mean(values == 0))


def fraction_genes_in_largest_module(evaluation: Evaluation) -> float:
    """Fraction of evaluated genes assigned to the largest module."""
    genes = _table(evaluation, "genes")
    sizes = _finite(_table(evaluation, "modules"), "n_genes")
    return float(sizes.max() / len(genes)) if len(genes) and sizes.size else np.nan


def number_meaningful_modules(evaluation: Evaluation, *, min_module_size: int = 10) -> int:
    """Number of modules at or above the meaningful-module size threshold."""
    return int(len(_meaningful(evaluation, min_module_size)))


def _species_balance(modules: pd.DataFrame) -> np.ndarray:
    mouse = _numeric(modules, _find_column(modules, "n_mouse"))
    human = _numeric(modules, _find_column(modules, "n_human"))
    total = mouse + human
    values = np.full(len(modules), np.nan)
    valid = np.isfinite(total) & (total > 0)
    values[valid] = 2.0 * np.minimum(mouse[valid], human[valid]) / total[valid]
    return values


def _make_balance_quantile(name: str, q: float):
    def function(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
        return _quantile(_species_balance(_meaningful(evaluation, min_module_size)), q)
    function.__name__ = name
    function.__doc__ = "Quantile of meaningful-module species balance (0 single-species, 1 equal)."
    return function


for _label, _q in zip(QUANTILE_LABELS, QUANTILES):
    _name = f"{_label}_module_species_balance"
    globals()[_name] = _make_balance_quantile(_name, _q)


def fraction_genes_in_mixed_meaningful_modules(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
    """Fraction of genes in meaningful modules containing both species."""
    genes = _table(evaluation, "genes")
    if not len(genes):
        return np.nan
    modules = _meaningful(evaluation, min_module_size)
    mouse = _numeric(modules, _find_column(modules, "n_mouse"))
    human = _numeric(modules, _find_column(modules, "n_human"))
    sizes = _numeric(modules, "n_genes")
    return float(np.sum(sizes[(mouse > 0) & (human > 0)]) / len(genes))


def _fraction_meaningful_with(evaluation: Evaluation, min_module_size: int, column: str) -> float:
    modules = _meaningful(evaluation, min_module_size)
    return float(np.mean(np.isfinite(_numeric(modules, column)))) if len(modules) else np.nan


def fraction_meaningful_modules_with_mouse_cohesion(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
    modules = _meaningful(evaluation, min_module_size)
    return _fraction_meaningful_with(evaluation, min_module_size, _cohesion_columns(modules)[0])


def fraction_meaningful_modules_with_human_cohesion(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
    modules = _meaningful(evaluation, min_module_size)
    return _fraction_meaningful_with(evaluation, min_module_size, _cohesion_columns(modules)[1])


def fraction_meaningful_modules_with_cross_species_cohesion(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
    modules = _meaningful(evaluation, min_module_size)
    return _fraction_meaningful_with(evaluation, min_module_size, _cohesion_columns(modules)[2])


def fraction_meaningful_modules_with_both_diversities(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
    modules = _meaningful(evaluation, min_module_size)
    mouse, human, _ = _diversity_columns(modules)
    valid = np.isfinite(_numeric(modules, mouse)) & np.isfinite(_numeric(modules, human))
    return float(np.mean(valid)) if len(modules) else np.nan


def worst_species_p10_gene_margin(evaluation: Evaluation) -> float:
    """Lower of the mouse and human 10th-percentile gene margins."""
    genes = _table(evaluation, "genes")
    values = []
    for species in ("mouse", "human"):
        subset = genes.loc[genes["species"].astype(str).str.casefold() == species]
        values.append(_quantile(_finite(subset, "gene_margin"), 0.10))
    return float(min(values)) if np.isfinite(values).all() else np.nan


def absolute_species_median_margin_difference(evaluation: Evaluation) -> float:
    """Absolute difference between mouse and human median gene margins."""
    genes = _table(evaluation, "genes")
    values = []
    for species in ("mouse", "human"):
        subset = genes.loc[genes["species"].astype(str).str.casefold() == species]
        values.append(_quantile(_finite(subset, "gene_margin"), 0.50))
    return float(abs(values[0] - values[1])) if np.isfinite(values).all() else np.nan


def _cohesion_contrast(evaluation: Evaluation, min_module_size: int) -> np.ndarray:
    modules, columns = _shared_cohesion_modules(evaluation, min_module_size)
    mouse, human, cross = (_numeric(modules, column) for column in columns)
    return cross - (mouse + human) / 2.0


def p10_cross_vs_within_cohesion_contrast(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
    """10th percentile of cross cohesion minus mean within-species cohesion."""
    return _quantile(_cohesion_contrast(evaluation, min_module_size), 0.10)


def median_cross_vs_within_cohesion_contrast(evaluation: Evaluation, *, min_module_size: int = 10) -> float:
    """Median cross cohesion minus mean within-species cohesion."""
    return _quantile(_cohesion_contrast(evaluation, min_module_size), 0.50)


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

    def calculate(self, evaluation: Evaluation, *, min_module_size: int = 10):
        if self.uses_meaningful_modules:
            return self.function(evaluation, min_module_size=min_module_size)
        return self.function(evaluation)


def _definition(name: str, category: str, direction: str, level: str, uses_modules: bool = False):
    function = globals()[name]
    return MetricDefinition(name, category, direction, level, function.__doc__ or "", function, uses_modules)


_definitions: list[MetricDefinition] = []
for _label in QUANTILE_LABELS:
    _definitions.append(_definition(f"{_label}_gene_margin", "separation", "maximize", "gene"))
_definitions.extend([
    _definition("fraction_negative_margin", "separation", "minimize", "gene"),
    _definition("worst_species_p10_gene_margin", "separation", "maximize", "gene"),
    _definition("absolute_species_median_margin_difference", "separation", "minimize", "gene"),
])
for _stem in ("cross_strength_fraction", "reciprocal_cross_fraction"):
    for _label in QUANTILE_LABELS:
        _definitions.append(_definition(f"{_label}_{_stem}", "integration", "target", "gene"))
_definitions.extend([
    _definition("cross_edge_coverage", "integration", "maximize", "gene"),
    _definition("reciprocal_gene_coverage", "integration", "maximize", "gene"),
    _definition("fraction_genes_in_mixed_meaningful_modules", "integration", "maximize", "gene", True),
])
for _family, _members in (
    ("cohesion", ("mouse", "human", "cross_species")),
    ("diversity", ("mouse", "human", "conserved")),
):
    for _member in _members:
        for _label in QUANTILE_LABELS:
            _definitions.append(_definition(f"{_label}_{_member}_{_family}", "module_quality", "maximize", "module", True))
_definitions.extend([
    _definition("fraction_meaningful_modules_with_mouse_cohesion", "module_quality", "maximize", "module", True),
    _definition("fraction_meaningful_modules_with_human_cohesion", "module_quality", "maximize", "module", True),
    _definition("fraction_meaningful_modules_with_cross_species_cohesion", "module_quality", "maximize", "module", True),
    _definition("fraction_meaningful_modules_with_both_diversities", "module_quality", "maximize", "module", True),
    _definition("p10_cross_vs_within_cohesion_contrast", "module_quality", "target", "module", True),
    _definition("median_cross_vs_within_cohesion_contrast", "module_quality", "target", "module", True),
])
for _label in QUANTILE_LABELS:
    _definitions.append(_definition(f"{_label}_module_species_balance", "integration", "maximize", "module", True))
_definitions.extend([
    _definition("fraction_isolated_genes", "module_structure", "minimize", "gene"),
    _definition("fraction_genes_in_largest_module", "module_structure", "target", "gene"),
    _definition("number_meaningful_modules", "module_structure", "target", "module", True),
])

METRIC_DEFINITIONS = tuple(_definitions)
METRIC_REGISTRY = {definition.name: definition for definition in METRIC_DEFINITIONS}


def metric_vector(evaluation: Evaluation, *, min_module_size: int = 10, metrics: Sequence[str] | None = None) -> pd.Series:
    """Calculate an ordered, named metric vector from one evaluation."""
    names = list(METRIC_REGISTRY) if metrics is None else list(metrics)
    unknown = [name for name in names if name not in METRIC_REGISTRY]
    if unknown:
        raise KeyError(f"unknown metrics: {unknown}")
    if len(names) != len(set(names)):
        raise ValueError("metrics contains duplicate names")
    result = pd.Series({
        name: METRIC_REGISTRY[name].calculate(evaluation, min_module_size=min_module_size)
        for name in names
    }, name="value")
    result.attrs["min_module_size"] = int(min_module_size)
    return result


def metric_metadata() -> pd.DataFrame:
    """Return the metric registry as a human-readable table."""
    return pd.DataFrame([{
        "metric": item.name,
        "category": item.category,
        "direction": item.direction,
        "level": item.level,
        "uses_meaningful_modules": item.uses_meaningful_modules,
        "description": item.description,
    } for item in METRIC_DEFINITIONS])


__all__ = [
    "METRIC_DEFINITIONS", "METRIC_REGISTRY", "MetricDefinition",
    "QUANTILES", "QUANTILE_LABELS", "metric_metadata", "metric_vector",
    *[definition.name for definition in METRIC_DEFINITIONS],
]
