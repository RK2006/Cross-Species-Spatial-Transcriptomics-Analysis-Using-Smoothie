"""Resumable Sobol exploration of the four graph hyperparameters."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import time
import traceback
from typing import Sequence

import matplotlib.pyplot as plt
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
    show: bool = False,
):
    """Plot selected raw indicators after display-only row normalization."""
    metrics = [metric for metric in metrics if metric in raw_metrics]
    if not metrics:
        raise ValueError("none of the requested heatmap metrics are available")
    common = raw_metrics.index.intersection(settings.index)
    raw = raw_metrics.loc[common, metrics]
    setting_values = settings.loc[common, list(PARAMETERS)]
    normalized = _robust_heatmap_values(raw, metrics)
    matrix = normalized.to_numpy(dtype=float).T

    order = np.arange(len(common))
    if cluster_runs and len(common) > 2:
        clustering_matrix = np.nan_to_num(normalized.to_numpy(dtype=float), nan=0.5)
        distances = pdist(clustering_matrix, metric="euclidean")
        if np.any(distances > 0):
            order = leaves_list(linkage(distances, method="average"))
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
    ax_metrics.set_xlabel(f"{len(common)} completed Sobol runs; columns clustered by raw metric profile")
    if len(common) <= 64:
        ax_metrics.set_xticks(range(len(common)), [str(common[index]) for index in order], rotation=90, fontsize=6)
    else:
        ax_metrics.set_xticks([])
    colorbar = fig.colorbar(image, cax=ax_color)
    colorbar.set_label("Within-metric robust position (Q10=0, Q90=1)")
    fig.suptitle("Raw clustering indicators across the Sobol search")

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
    "prepare_neighbor_cache", "run_wide_sobol_search", "save_neighbor_cache",
]
