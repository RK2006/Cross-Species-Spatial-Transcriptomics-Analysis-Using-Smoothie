"""Resumable Sobol exploration of the four graph hyperparameters."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
from itertools import combinations
import json
import math
from pathlib import Path
import time
import traceback
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from .artifacts import gene_order_hash
from .evaluation import evaluate_joint_modules
from .metrics import metric_vector
from .objective import educated_full_score_spec, score_objective
from .search import cache_neighbors, cached_graph, diagnostics


PARAMETERS = ("k_intra", "k_inter", "cutoff_intra", "cutoff_inter")
DEFAULT_PARAMETER_RANGES = {
    "k_intra": (10, 500),
    "k_inter": (10, 500),
    "cutoff_intra": (0.80, 0.98),
    "cutoff_inter": (0.50, 0.85),
}
SOBOL_CHECKPOINTS = (128, 256, 512)

KEY_HEATMAP_METRICS = (
    "p10_gene_margin",
    "median_gene_margin",
    "fraction_negative_margin",
    "worst_species_p10_gene_margin",
    "median_cross_strength_fraction",
    "cross_edge_coverage",
    "reciprocal_gene_coverage",
    "p25_module_species_balance",
    "fraction_genes_in_mixed_meaningful_modules",
    "p10_cross_species_cohesion",
    "median_cross_species_cohesion",
    "p10_conserved_diversity",
    "median_conserved_diversity",
    "median_cross_vs_within_cohesion_contrast",
    "fraction_meaningful_modules_with_cross_species_cohesion",
    "fraction_meaningful_modules_with_both_diversities",
    "fraction_isolated_genes",
    "fraction_genes_in_largest_module",
    "number_meaningful_modules",
)

METRIC_DIRECTIONS = {
    "fraction_negative_margin": "↓",
    "absolute_species_median_margin_difference": "↓",
    "fraction_isolated_genes": "↓",
    "fraction_genes_in_largest_module": "↔",
    "number_meaningful_modules": "↔",
    "median_cross_strength_fraction": "↔",
    "median_cross_vs_within_cohesion_contrast": "↔",
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def generate_sobol_settings(
    parameter_ranges=DEFAULT_PARAMETER_RANGES,
    *,
    n_settings: int = 128,
    seed: int = 0,
    k_sampling: str = "log",
) -> pd.DataFrame:
    """Generate unique four-parameter settings from a scrambled Sobol sequence."""
    if n_settings < 2 or n_settings & (n_settings - 1):
        raise ValueError("n_settings must be a power of two")
    if k_sampling not in {"linear", "log"}:
        raise ValueError("k_sampling must be 'linear' or 'log'")
    bounds = {}
    for name in PARAMETERS:
        if name not in parameter_ranges:
            raise ValueError(f"parameter_ranges is missing {name}")
        low, high = map(float, parameter_ranges[name])
        if not low < high:
            raise ValueError(f"{name} must have minimum < maximum")
        if name.startswith("k_") and low < 1:
            raise ValueError(f"{name} must be positive")
        if name.startswith("cutoff_") and not (0 < low < high <= 1):
            raise ValueError(f"{name} must lie in (0, 1]")
        bounds[name] = (low, high)

    draw_count = max(32, n_settings * 8)
    points = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(
        int(math.ceil(math.log2(draw_count)))
    )
    rows, seen = [], set()
    for sobol_index, point in enumerate(points):
        values = {}
        for coordinate, name in enumerate(PARAMETERS):
            low, high = bounds[name]
            if name.startswith("k_"):
                value = (
                    np.exp(np.log(low) + point[coordinate] * (np.log(high) - np.log(low)))
                    if k_sampling == "log"
                    else low + point[coordinate] * (high - low)
                )
                values[name] = int(np.clip(np.rint(value), np.ceil(low), np.floor(high)))
            else:
                values[name] = float(low + point[coordinate] * (high - low))
        identity = (
            values["k_intra"], values["k_inter"],
            round(values["cutoff_intra"], 10),
            round(values["cutoff_inter"], 10),
        )
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({"sobol_id": len(rows), "sobol_index": sobol_index, **values})
        if len(rows) == n_settings:
            break
    if len(rows) != n_settings:
        raise RuntimeError("unable to generate enough unique Sobol settings")
    return pd.DataFrame(rows)


def save_neighbor_cache(cache, output_dir, *, genes, species) -> Path:
    """Persist the four compact cache arrays and gene-order metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "n": int(cache["n"]),
        "split": int(cache["split"]),
        "max_intra": int(cache["intra"][0].shape[1]),
        "max_inter": int(cache["inter"][0].shape[1]),
        "gene_order_sha256": gene_order_hash(genes, species),
    }
    for kind in ("intra", "inter"):
        np.save(output_dir / f"{kind}_indices.npy", np.asarray(cache[kind][0], dtype=np.int32))
        np.save(output_dir / f"{kind}_weights.npy", np.asarray(cache[kind][1], dtype=np.float32))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return output_dir


def load_neighbor_cache(output_dir, *, genes, species, mmap_mode="r"):
    """Load and validate a neighbor cache, optionally as memory-mapped arrays."""
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing neighbor-cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported neighbor-cache schema")
    if int(metadata["n"]) != len(genes) or len(genes) != len(species):
        raise ValueError("neighbor cache has the wrong number of genes")
    if metadata["gene_order_sha256"] != gene_order_hash(genes, species):
        raise ValueError("neighbor cache uses a different gene/species order")
    cache = {"n": int(metadata["n"]), "split": int(metadata["split"])}
    for kind in ("intra", "inter"):
        indices = np.load(output_dir / f"{kind}_indices.npy", mmap_mode=mmap_mode)
        weights = np.load(output_dir / f"{kind}_weights.npy", mmap_mode=mmap_mode)
        if indices.shape != weights.shape or indices.shape[0] != cache["n"]:
            raise ValueError(f"invalid {kind} cache array shapes")
        cache[kind] = (indices, weights)
    return cache


def prepare_neighbor_cache(
    C,
    split: int,
    *,
    genes,
    species,
    max_intra: int,
    max_inter: int,
    cache_dir,
    block: int = 256,
):
    """Load a compatible persistent cache or build and save it once."""
    cache_dir = Path(cache_dir)
    if (cache_dir / "metadata.json").exists():
        cache = load_neighbor_cache(cache_dir, genes=genes, species=species)
        if cache["intra"][0].shape[1] < max_intra or cache["inter"][0].shape[1] < max_inter:
            raise ValueError("saved cache budgets are smaller than the requested search")
        return cache
    cache = cache_neighbors(C, split, max_intra=max_intra, max_inter=max_inter, block=block)
    save_neighbor_cache(cache, cache_dir, genes=genes, species=species)
    del cache
    gc.collect()
    return load_neighbor_cache(cache_dir, genes=genes, species=species)


def order_settings_fast_first(
    settings: pd.DataFrame,
    cache,
    *,
    stage_size: int = 128,
    sampled_genes: int = 1024,
) -> pd.DataFrame:
    """Estimate graph size and sort cheaply within balanced Sobol stages."""
    if stage_size < 1:
        raise ValueError("stage_size must be positive")
    n = int(cache["n"])
    rows = np.unique(np.linspace(0, n - 1, min(sampled_genes, n), dtype=int))
    estimates = []
    for setting in settings.itertuples(index=False):
        total = 0
        for kind in ("intra", "inter"):
            k = int(getattr(setting, f"k_{kind}"))
            cutoff = float(getattr(setting, f"cutoff_{kind}"))
            weights = cache[kind][1][rows, :k]
            total += int(np.count_nonzero(np.isfinite(weights) & (weights >= cutoff)))
        estimates.append(total / len(rows))
    ordered = settings.copy()
    ordered["estimated_directed_proposals_per_gene"] = estimates
    ordered["stage"] = ordered["sobol_id"] // stage_size + 1
    ordered = ordered.sort_values(
        ["stage", "estimated_directed_proposals_per_gene", "sobol_id"], kind="stable"
    ).reset_index(drop=True)
    ordered.insert(0, "execution_order", np.arange(len(ordered), dtype=int))
    return ordered


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2))
    temporary.replace(path)


def collect_search_results(output_dir) -> dict[str, pd.DataFrame]:
    """Rebuild compact tables from per-run JSON checkpoints."""
    output_dir = Path(output_dir)
    payloads = [json.loads(path.read_text()) for path in sorted((output_dir / "runs").glob("sobol_*.json"))]
    completed = [item for item in payloads if item.get("status") == "complete"]
    summary_rows, metric_rows, category_rows, family_rows, desirability_rows = [], [], [], [], []
    for item in completed:
        key = int(item["sobol_id"])
        common = {"sobol_id": key, "execution_order": item["execution_order"], **item["parameters"]}
        summary_rows.append({
            **common,
            "runtime_seconds": item["runtime_seconds"],
            "final_score": item["score"]["final_score"],
            "feasible": item["score"]["feasible"],
            "violations": "; ".join(item["score"]["violations"]),
            **item["diagnostics"],
        })
        metric_rows.append({"sobol_id": key, **item["metrics"]})
        category_rows.append({"sobol_id": key, **item["score"]["categories"]})
        family_rows.append({"sobol_id": key, **item["score"]["families"]})
        desirability_rows.append({"sobol_id": key, **item["score"]["desirabilities"]})

    def frame(rows):
        return pd.DataFrame(rows).set_index("sobol_id").sort_index() if rows else pd.DataFrame()

    tables = {
        "summary": frame(summary_rows),
        "raw_metrics": frame(metric_rows),
        "category_scores": frame(category_rows),
        "family_scores": frame(family_rows),
        "desirabilities": frame(desirability_rows),
        "status": pd.DataFrame(payloads),
    }
    for name, table in tables.items():
        if name != "status" and not table.empty:
            table.to_csv(output_dir / f"{name}.csv")
    if payloads:
        pd.DataFrame([{
            "sobol_id": item["sobol_id"],
            "execution_order": item["execution_order"],
            "status": item["status"],
            "runtime_seconds": item.get("runtime_seconds"),
            "error": item.get("error"),
        } for item in payloads]).sort_values("execution_order").to_csv(
            output_dir / "run_status.csv", index=False
        )
    return tables


def _robust_heatmap_values(raw_metrics: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    values = raw_metrics.loc[:, metrics].apply(pd.to_numeric, errors="coerce")
    normalized = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    for metric in values:
        finite = values[metric].dropna()
        if finite.empty:
            normalized[metric] = np.nan
            continue
        low, high = finite.quantile([0.10, 0.90])
        normalized[metric] = 0.5 if high <= low else ((values[metric] - low) / (high - low)).clip(0, 1)
    return normalized


def plot_metric_heatmap(
    raw_metrics: pd.DataFrame,
    settings: pd.DataFrame,
    output_path,
    *,
    metrics: Sequence[str] = KEY_HEATMAP_METRICS,
    cluster_runs: bool = True,
    sort_by: str | Sequence[str] | None = None,
    ascending: bool | Sequence[bool] = True,
    title: str | None = None,
    show: bool = False,
):
    """Plot selected raw indicators after display-only row normalization.

    ``sort_by`` may name columns in either ``settings`` or ``raw_metrics``.
    It takes precedence over hierarchical clustering and can be a sequence for
    lexicographic ordering.
    """
    metrics = [metric for metric in metrics if metric in raw_metrics]
    if not metrics:
        raise ValueError("none of the requested heatmap metrics are available")
    common = raw_metrics.index.intersection(settings.index)
    raw = raw_metrics.loc[common, metrics]
    setting_values = settings.loc[common, list(PARAMETERS)]
    normalized = _robust_heatmap_values(raw, metrics)
    matrix = normalized.to_numpy(dtype=float).T

    order = np.arange(len(common))
    order_description = "Sobol order"
    if sort_by is not None:
        keys = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        if not keys:
            raise ValueError("sort_by must contain at least one column")
        ordering = pd.DataFrame(index=common)
        for key in keys:
            if key in settings:
                ordering[key] = settings.loc[common, key]
            elif key in raw_metrics:
                ordering[key] = raw_metrics.loc[common, key]
            else:
                raise ValueError(f"sort column is unavailable: {key}")
        sorted_index = ordering.sort_values(
            keys, ascending=ascending, na_position="last", kind="stable"
        ).index
        order = common.get_indexer(sorted_index)
        directions = (
            [ascending] * len(keys) if isinstance(ascending, (bool, np.bool_))
            else list(ascending)
        )
        if len(directions) != len(keys):
            raise ValueError("ascending must be one bool or match sort_by")
        order_description = "sorted by " + ", ".join(
            f"{key} ({'low→high' if direction else 'high→low'})"
            for key, direction in zip(keys, directions)
        )
    elif cluster_runs and len(common) > 2:
        clustering_matrix = np.nan_to_num(normalized.to_numpy(dtype=float), nan=0.5)
        distances = pdist(clustering_matrix, metric="euclidean")
        if np.any(distances > 0):
            order = leaves_list(linkage(distances, method="average"))
            order_description = "clustered by raw metric profile"
    matrix = matrix[:, order]
    setting_values = setting_values.iloc[order]

    parameter_matrix = np.empty((len(PARAMETERS), len(setting_values)))
    for row, parameter in enumerate(PARAMETERS):
        values = setting_values[parameter].to_numpy(dtype=float)
        if parameter.startswith("k_"):
            values = np.log(values)
        low, high = np.nanmin(values), np.nanmax(values)
        parameter_matrix[row] = 0.5 if high <= low else (values - low) / (high - low)

    width = max(14, min(30, len(common) / 12))
    height = max(9, len(metrics) * 0.38 + 3)
    fig = plt.figure(figsize=(width, height), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, len(metrics)), width_ratios=(1, 0.025))
    ax_parameters = fig.add_subplot(grid[0, 0])
    ax_metrics = fig.add_subplot(grid[1, 0])
    ax_color = fig.add_subplot(grid[1, 1])

    ax_parameters.imshow(parameter_matrix, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax_parameters.set_yticks(range(len(PARAMETERS)), [name.replace("_", " ") for name in PARAMETERS])
    ax_parameters.set_xticks([])
    ax_parameters.set_title("Hyperparameter annotations (normalized within search range)")

    image = ax_metrics.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    labels = [f"{METRIC_DIRECTIONS.get(metric, '↑')}  {metric.replace('_', ' ')}" for metric in metrics]
    ax_metrics.set_yticks(range(len(metrics)), labels)
    ax_metrics.set_xlabel(f"{len(common)} completed Sobol runs; columns {order_description}")
    if len(common) <= 64:
        ax_metrics.set_xticks(range(len(common)), [str(common[index]) for index in order], rotation=90, fontsize=6)
    else:
        ax_metrics.set_xticks([])
    colorbar = fig.colorbar(image, cax=ax_color)
    colorbar.set_label("Within-metric robust position (Q10=0, Q90=1)")
    fig.suptitle(title or "Raw clustering indicators across the Sobol search")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    normalized.to_csv(output_path.with_suffix(".normalized.csv"))
    pd.DataFrame({
        "heatmap_column": np.arange(len(order), dtype=int),
        "sobol_id": np.asarray(common)[order],
    }).to_csv(output_path.with_suffix(".column_order.csv"), index=False)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, normalized


def plot_pairwise_parameter_landscapes(
    summary: pd.DataFrame,
    output_path,
    *,
    value: str = "final_score",
    bins: int = 6,
    aggregation: str = "median",
    minimum_bin_count: int = 3,
    show: bool = False,
):
    """Plot all six two-parameter landscapes, marginalizing over the other two.

    The two neighbor-count axes are binned on a logarithmic scale. Cutoffs are
    binned linearly. Each cell shows the requested aggregate and the number of
    runs contributing to it; sparsely occupied cells are left blank.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    required = [*PARAMETERS, value]
    missing = [column for column in required if column not in summary]
    if missing:
        raise ValueError(f"summary is missing columns: {missing}")
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")

    frame = summary.loc[:, required].apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ValueError("no finite landscape values are available")
    binned = pd.DataFrame(index=frame.index)
    labels = {}
    for parameter in PARAMETERS:
        values = frame[parameter].to_numpy(dtype=float)
        if parameter.startswith("k_"):
            edges = np.geomspace(values.min(), values.max(), bins + 1)
            edges[0] = np.nextafter(edges[0], -np.inf)
            labels[parameter] = [f"{value:.0f}" for value in np.sqrt(edges[:-1] * edges[1:])]
        else:
            edges = np.linspace(values.min(), values.max(), bins + 1)
            edges[0] = np.nextafter(edges[0], -np.inf)
            labels[parameter] = [f"{value:.3f}" for value in (edges[:-1] + edges[1:]) / 2]
        binned[parameter] = pd.cut(values, edges, labels=False, include_lowest=True)
    binned[value] = frame[value]

    finite_values = frame[value].to_numpy(dtype=float)
    color_low = min(0.0, float(np.nanquantile(finite_values, 0.02)))
    color_high = float(np.nanquantile(finite_values, 0.98))
    if color_high <= color_low:
        color_high = float(np.nanmax(finite_values))
    pairs = list(combinations(PARAMETERS, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=180, constrained_layout=True)
    image = None
    for ax, (vertical, horizontal) in zip(axes.flat, pairs):
        grouped = binned.groupby([vertical, horizontal], observed=True)[value]
        aggregate = getattr(grouped, aggregation)().unstack()
        counts = grouped.count().unstack()
        aggregate = aggregate.reindex(index=range(bins), columns=range(bins))
        counts = counts.reindex(index=range(bins), columns=range(bins)).fillna(0)
        matrix = aggregate.to_numpy(dtype=float)
        matrix[counts.to_numpy() < minimum_bin_count] = np.nan
        image = ax.imshow(
            matrix, origin="lower", aspect="auto", cmap="viridis",
            vmin=color_low, vmax=color_high, interpolation="nearest",
        )
        ax.set_xticks(range(bins), labels[horizontal], rotation=45, ha="right")
        ax.set_yticks(range(bins), labels[vertical])
        ax.set_xlabel(horizontal.replace("_", " "))
        ax.set_ylabel(vertical.replace("_", " "))
        ax.set_title(f"{aggregation.title()} {value.replace('_', ' ')}")
        for row in range(bins):
            for column in range(bins):
                count = int(counts.iloc[row, column])
                result = matrix[row, column]
                if np.isfinite(result):
                    color = "black" if result > (color_low + color_high) / 2 else "white"
                    ax.text(
                        column, row, f"{result:.3f}\nn={count}",
                        ha="center", va="center", fontsize=6, color=color,
                    )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
        colorbar.set_label(f"{aggregation.title()} {value.replace('_', ' ')}")
    fig.suptitle(
        f"Pairwise hyperparameter landscapes ({aggregation} over the other two parameters)",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _parameter_bin_summary(
    parameter_values: pd.Series,
    metric_values: pd.Series,
    *,
    bins: int,
) -> pd.DataFrame:
    """Return display-only median and IQR guides for a one-parameter plot."""
    finite = np.isfinite(parameter_values) & np.isfinite(metric_values)
    x = parameter_values.loc[finite].astype(float)
    y = metric_values.loc[finite].astype(float)
    if x.empty or x.min() >= x.max():
        return pd.DataFrame(columns=["center", "p25", "median", "p75", "count"])
    if parameter_values.name.startswith("k_"):
        edges = np.geomspace(x.min(), x.max(), bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(x.min(), x.max(), bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
    edges[0] = np.nextafter(edges[0], -np.inf)
    groups = pd.cut(x, edges, labels=False, include_lowest=True)
    values = pd.DataFrame({"group": groups, "value": y}).groupby("group", observed=True)["value"]
    result = values.agg(
        p25=lambda values: values.quantile(0.25),
        median="median",
        p75=lambda values: values.quantile(0.75),
        count="count",
    ).reindex(range(bins))
    result.insert(0, "center", centers)
    return result


def plot_indicator_pairwise_scatter(
    summary: pd.DataFrame,
    metric_values: pd.Series,
    output_path,
    *,
    metric_name: str | None = None,
    show: bool = False,
):
    """Plot all six exact pairwise parameter views, colored by one indicator."""
    metric_name = metric_name or metric_values.name or "indicator"
    frame = summary.loc[:, list(PARAMETERS)].copy()
    frame[metric_name] = pd.to_numeric(metric_values, errors="coerce")
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ValueError(f"no finite values are available for {metric_name}")
    color_low, color_high = frame[metric_name].min(), frame[metric_name].max()
    if color_high <= color_low:
        color_low -= 0.5
        color_high += 0.5

    pairs = list(combinations(PARAMETERS, 2))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=180, constrained_layout=True)
    image = None
    for ax, (horizontal, vertical) in zip(axes.flat, pairs):
        image = ax.scatter(
            frame[horizontal], frame[vertical], c=frame[metric_name],
            cmap="viridis", vmin=color_low, vmax=color_high,
            s=20, alpha=0.82, linewidths=0, rasterized=True,
        )
        if horizontal.startswith("k_"):
            ax.set_xscale("log")
        if vertical.startswith("k_"):
            ax.set_yscale("log")
        ax.set_xlabel(horizontal.replace("_", " "))
        ax.set_ylabel(vertical.replace("_", " "))
        ax.grid(alpha=0.15, linewidth=0.6)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
        colorbar.set_label(metric_name.replace("_", " "))
    direction = METRIC_DIRECTIONS.get(metric_name, "↑")
    fig.suptitle(
        f"{metric_name.replace('_', ' ')} ({direction}): exact Sobol runs across parameter pairs\n"
        f"n={len(frame)}; color shows the raw indicator value",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_indicator_main_effects(
    summary: pd.DataFrame,
    metric_values: pd.Series,
    output_path,
    *,
    metric_name: str | None = None,
    guide_bins: int = 12,
    show: bool = False,
):
    """Plot one indicator against each parameter without discarding raw points."""
    metric_name = metric_name or metric_values.name or "indicator"
    frame = summary.loc[:, list(PARAMETERS)].copy()
    frame[metric_name] = pd.to_numeric(metric_values, errors="coerce")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180, constrained_layout=True)
    for ax, parameter in zip(axes.flat, PARAMETERS):
        finite = frame[[parameter, metric_name]].dropna()
        ax.scatter(
            finite[parameter], finite[metric_name],
            s=15, alpha=0.38, color="#3366a6", linewidths=0, rasterized=True,
            label="individual run",
        )
        guide = _parameter_bin_summary(
            finite[parameter].rename(parameter), finite[metric_name], bins=guide_bins,
        ).dropna(subset=["median"])
        if not guide.empty:
            ax.fill_between(
                guide["center"].to_numpy(), guide["p25"].to_numpy(),
                guide["p75"].to_numpy(), color="#e69500", alpha=0.20,
                label="binned IQR guide",
            )
            ax.plot(
                guide["center"], guide["median"], color="#c56b00", linewidth=2,
                marker="o", markersize=3, label="binned median guide",
            )
        if parameter.startswith("k_"):
            ax.set_xscale("log")
        correlation = finite[parameter].corr(finite[metric_name], method="spearman")
        ax.set_title(f"Spearman ρ = {correlation:.2f}")
        ax.set_xlabel(parameter.replace("_", " "))
        ax.set_ylabel(metric_name.replace("_", " "))
        ax.grid(alpha=0.15, linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    direction = METRIC_DIRECTIONS.get(metric_name, "↑")
    fig.suptitle(
        f"{metric_name.replace('_', ' ')} ({direction}) versus each hyperparameter\n"
        "Points are exact runs; lines and bands are display-only median/IQR guides",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_important_indicator_relationships(
    summary: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    output_dir,
    *,
    metrics: Sequence[str] = KEY_HEATMAP_METRICS,
    guide_bins: int = 12,
    combined_pdf_name: str = "important_indicator_parameter_plots.pdf",
    show: bool = False,
) -> pd.DataFrame:
    """Create pairwise and main-effect figures for every selected indicator."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available = [metric for metric in metrics if metric in raw_metrics]
    if not available:
        raise ValueError("none of the requested indicators are available")
    manifest_rows = []
    pdf_path = output_dir / combined_pdf_name
    with PdfPages(pdf_path) as pdf:
        for metric in available:
            pairwise_path = output_dir / f"{metric}_pairwise.png"
            effects_path = output_dir / f"{metric}_main_effects.png"
            pairwise = plot_indicator_pairwise_scatter(
                summary, raw_metrics[metric], pairwise_path,
                metric_name=metric, show=show,
            )
            pdf.savefig(pairwise, bbox_inches="tight")
            effects = plot_indicator_main_effects(
                summary, raw_metrics[metric], effects_path,
                metric_name=metric, guide_bins=guide_bins, show=show,
            )
            pdf.savefig(effects, bbox_inches="tight")
            if not show:
                plt.close(pairwise)
                plt.close(effects)
            finite = pd.to_numeric(raw_metrics[metric], errors="coerce").dropna()
            manifest_rows.append({
                "metric": metric,
                "direction": METRIC_DIRECTIONS.get(metric, "↑"),
                "n_finite": len(finite),
                "minimum": finite.min(),
                "median": finite.median(),
                "maximum": finite.max(),
                "pairwise_plot": pairwise_path.name,
                "main_effects_plot": effects_path.name,
            })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "plot_manifest.csv", index=False)
    return manifest


def run_wide_sobol_search(
    prof,
    C,
    *,
    expression_by_species,
    output_dir,
    cache_dir,
    parameter_ranges=DEFAULT_PARAMETER_RANGES,
    n_settings: int = 512,
    stage_size: int = 128,
    resolution: float = 1.0,
    leiden_seed: int = 0,
    sobol_seed: int = 0,
    meaningful_module_size: int = 10,
    matrix_block: int = 512,
    continue_on_error: bool = True,
    show_heatmap: bool = False,
):
    """Run, checkpoint, resume, and summarize a wide no-PCA Sobol search."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    if C.shape != (len(prof.genes), len(prof.genes)):
        raise ValueError("C and prof describe different gene universes")
    if n_settings % stage_size:
        raise ValueError("n_settings must be divisible by stage_size")

    configuration = {
        "schema_version": 1,
        "representation": "original_second_order_pcc",
        "n_settings": int(n_settings),
        "stage_size": int(stage_size),
        "resolution": float(resolution),
        "leiden_seed": int(leiden_seed),
        "sobol_seed": int(sobol_seed),
        "meaningful_module_size": int(meaningful_module_size),
        "parameter_ranges": _jsonable(dict(parameter_ranges)),
        "gene_order_sha256": gene_order_hash(prof.genes, prof.species),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    configuration_path = output_dir / "configuration.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text())
        comparable = {key: value for key, value in configuration.items() if key != "created_utc"}
        existing_comparable = {key: value for key, value in existing.items() if key != "created_utc"}
        if existing_comparable != comparable:
            raise ValueError("existing output directory has an incompatible configuration")
    else:
        configuration_path.write_text(json.dumps(configuration, indent=2))

    design_path = output_dir / "sobol_design.csv"
    design = generate_sobol_settings(parameter_ranges, n_settings=n_settings, seed=sobol_seed, k_sampling="log")
    cache = prepare_neighbor_cache(
        C, int(prof.meta["width_a"]), genes=prof.genes, species=prof.species,
        max_intra=int(math.ceil(parameter_ranges["k_intra"][1])),
        max_inter=int(math.ceil(parameter_ranges["k_inter"][1])), cache_dir=cache_dir,
    )
    ordered = order_settings_fast_first(design, cache, stage_size=stage_size)
    if design_path.exists():
        saved = pd.read_csv(design_path)
        compare = ["execution_order", "sobol_id", *PARAMETERS]
        if saved.shape != ordered.shape or not np.allclose(
            saved[compare].to_numpy(dtype=float),
            ordered[compare].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("saved Sobol design differs from regenerated design")
    else:
        ordered.to_csv(design_path, index=False)

    score_specification = educated_full_score_spec()
    score_specification.rules_frame().to_csv(output_dir / "provisional_scoring_rules.csv", index=False)

    for setting in ordered.itertuples(index=False):
        run_name = f"sobol_{int(setting.sobol_id):04d}"
        checkpoint = output_dir / "runs" / f"{run_name}.json"
        if checkpoint.exists() and json.loads(checkpoint.read_text()).get("status") == "complete":
            continue
        parameters = {
            "k_intra": int(setting.k_intra),
            "k_inter": int(setting.k_inter),
            "cutoff_intra": float(setting.cutoff_intra),
            "cutoff_inter": float(setting.cutoff_inter),
        }
        started = time.perf_counter()
        graph = labels = evaluation = None
        try:
            graph = cached_graph(cache, prof.genes, prof.species, **parameters)
            labels = graph.leiden(resolution=resolution, seed=leiden_seed, verbose=False)
            evaluation = evaluate_joint_modules(
                labels, C, graph, prof,
                expression_by_species=expression_by_species,
                neighbor_cache=cache,
                k_inter=parameters["k_inter"], cutoff_inter=parameters["cutoff_inter"],
                min_module_size=2, return_module_scores=False,
                make_plots=False, show_plots=False, matrix_block=matrix_block,
            )
            metrics = metric_vector(evaluation, min_module_size=meaningful_module_size)
            try:
                score = score_objective(metrics, score_specification)
                score_payload = {
                    "final_score": score.final_score,
                    "feasible": score.feasible,
                    "violations": list(score.violations),
                    "categories": score.category_scores.to_dict(),
                    "families": {
                        f"{category}__{family}": value
                        for (category, family), value in score.family_scores.items()
                    },
                    "desirabilities": score.desirabilities.to_dict(),
                }
            except ValueError as scoring_error:
                # Raw indicators remain valuable when a provisional score is
                # undefined, for example when no module qualifies for diversity.
                score_payload = {
                    "final_score": np.nan,
                    "feasible": False,
                    "violations": [f"scoring undefined: {scoring_error}"],
                    "categories": {},
                    "families": {},
                    "desirabilities": {},
                }
            graph_diagnostics = diagnostics(graph, labels)
            np.savez_compressed(output_dir / "labels" / f"{run_name}.npz", labels=np.asarray(labels, dtype=np.int32))
            payload = {
                "status": "complete",
                "sobol_id": int(setting.sobol_id),
                "sobol_index": int(setting.sobol_index),
                "execution_order": int(setting.execution_order),
                "stage": int(setting.stage),
                "estimated_directed_proposals_per_gene": float(setting.estimated_directed_proposals_per_gene),
                "parameters": parameters,
                "runtime_seconds": time.perf_counter() - started,
                "diagnostics": graph_diagnostics,
                "metrics": metrics.to_dict(),
                "score": score_payload,
            }
            _atomic_json(checkpoint, payload)
            print(
                f"[{int(setting.execution_order) + 1}/{len(ordered)}] {run_name} "
                f"complete in {payload['runtime_seconds']:.1f}s",
                flush=True,
            )
        except Exception as error:
            payload = {
                "status": "error", "sobol_id": int(setting.sobol_id),
                "sobol_index": int(setting.sobol_index),
                "execution_order": int(setting.execution_order),
                "stage": int(setting.stage), "parameters": parameters,
                "runtime_seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _atomic_json(checkpoint, payload)
            print(f"{run_name} failed: {payload['error']}", flush=True)
            if not continue_on_error:
                raise
        finally:
            plt.close("all")
            del evaluation, labels, graph
            gc.collect()

        if (int(setting.execution_order) + 1) % stage_size == 0:
            stage_tables = collect_search_results(output_dir)
            if not stage_tables["raw_metrics"].empty:
                completed = len(stage_tables["raw_metrics"])
                plot_metric_heatmap(
                    stage_tables["raw_metrics"], ordered.set_index("sobol_id"),
                    output_dir / f"key_indicator_heatmap_{completed:04d}_runs.png",
                    show=False,
                )

    tables = collect_search_results(output_dir)
    if not tables["raw_metrics"].empty:
        settings = ordered.set_index("sobol_id")
        plot_metric_heatmap(
            tables["raw_metrics"], settings,
            output_dir / "key_indicator_heatmap.png", show=show_heatmap,
        )
    return {**tables, "design": ordered, "cache": cache, "output_dir": str(output_dir)}


__all__ = [
    "DEFAULT_PARAMETER_RANGES", "KEY_HEATMAP_METRICS", "PARAMETERS",
    "SOBOL_CHECKPOINTS", "collect_search_results", "generate_sobol_settings",
    "load_neighbor_cache", "order_settings_fast_first", "plot_metric_heatmap",
    "plot_important_indicator_relationships", "plot_indicator_main_effects",
    "plot_indicator_pairwise_scatter",
    "plot_pairwise_parameter_landscapes",
    "prepare_neighbor_cache", "run_wide_sobol_search", "save_neighbor_cache",
]
