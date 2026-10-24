"""PCA diagnostics, interpretation, and quasi-random clustering experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import evaluate_clusterings, evaluate_joint_modules
from .representation import (
    ANALYSIS_FIGURE_DPI,
    ANALYSIS_PLOT_FORMAT,
    ANALYSIS_SAVE_DPI,
    ANALYSIS_WIDE_FIGSIZE,
    REPRESENTATION_COLORS,
    pca_pearson_embedding,
    plot_edges,
    save_analysis_figures,
    similarity_from_unit_rows,
    style_analysis_axes,
)
from .search import cache_neighbors, cached_graph



def reference_cohesion_table(
    labels, C_reference, species, *, min_module_size=1
):
    """Per-module cohesion evaluated on one shared reference similarity matrix."""
    import numpy as np
    import pandas as pd

    labels = np.asarray(labels)
    species = np.asarray(species).astype(str)
    if C_reference.shape != (len(labels), len(labels)) or len(species) != len(labels):
        raise ValueError("labels, species, and C_reference must describe the same genes")
    species_values = list(dict.fromkeys(species.tolist()))
    if len(species_values) != 2:
        raise ValueError("reference cohesion requires exactly two species")
    lookup = {value.lower(): value for value in species_values}
    if {"mouse", "human"}.issubset(lookup):
        first_name, second_name = lookup["mouse"], lookup["human"]
    else:
        first_name, second_name = species_values

    def off_diagonal_mean(indices):
        if len(indices) < 2:
            return np.nan
        matrix = C_reference[np.ix_(indices, indices)]
        finite = np.isfinite(matrix)
        diagonal = np.diag(matrix)
        count = finite.sum() - np.isfinite(diagonal).sum()
        return (
            float((np.nansum(matrix) - np.nansum(diagonal)) / count)
            if count else np.nan
        )

    rows = []
    for module in np.unique(labels):
        members = np.flatnonzero(labels == module)
        if len(members) < min_module_size:
            continue
        first = members[species[members] == first_name]
        second = members[species[members] == second_name]
        cross = C_reference[np.ix_(first, second)]
        rows.append({
            "module": module,
            "reference_cohesion_all": off_diagonal_mean(members),
            f"reference_cohesion_{first_name}_{first_name}": off_diagonal_mean(first),
            f"reference_cohesion_{second_name}_{second_name}": off_diagonal_mean(second),
            "reference_cohesion_cross_species": (
                float(np.nanmean(cross)) if cross.size else np.nan
            ),
        })
    return pd.DataFrame(rows)


def run_pca_pilot(
    prof,
    C_reference,
    *,
    reference_neighbor_cache=None,
    expression_by_species=None,
    experiment_dir=None,
    baseline_name="run_049",
    dimensions=(50, 100, 200, 500),
    resolution=1.0,
    leiden_seed=0,
    neighbor_k=50,
    sampled_pairs_per_type=100_000,
    random_state=0,
    matrix_block=512,
    min_module_size=10,
    output_dir=None,
    show_plots=True,
):
    """Run a nested PCA screen using a saved no-PCA clustering as reference.

    The no-PCA baseline is loaded from disk and is never reclustered. One PCA
    is fitted at ``max(dimensions)``; smaller representations use prefixes of
    those scores. PCA similarity is Pearson correlation between centered score
    vectors. For each dimension the function measures similarity and
    neighbor preservation, constructs the baseline hyperparameter graph, runs
    Leiden, evaluates the modules, saves the result, and releases the dense
    PCA similarity matrix before continuing.
    """
    from datetime import datetime
    from itertools import combinations
    from pathlib import Path
    import gc
    import json
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.stats import pearsonr, spearmanr
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score

    dimensions = tuple(sorted({int(value) for value in dimensions}))
    if not dimensions or dimensions[0] < 1:
        raise ValueError("dimensions must contain positive integers")
    if neighbor_k < 1 or sampled_pairs_per_type < 100:
        raise ValueError("neighbor_k must be positive and use at least 100 sampled pairs")
    if matrix_block < 1 or min_module_size < 2:
        raise ValueError("matrix_block must be positive and min_module_size at least 2")

    C_reference = np.asarray(C_reference)
    n_genes = len(prof.genes)
    if C_reference.shape != (n_genes, n_genes):
        raise ValueError("C_reference and prof must describe the same genes")
    max_components = max(dimensions)
    profile_feature_count = int(prof.z.shape[0])
    if max_components > min(n_genes, profile_feature_count):
        raise ValueError(
            f"maximum PCA dimension {max_components} exceeds the available "
            f"rank {min(n_genes, profile_feature_count)}"
        )

    def find_experiment(directory):
        if directory is not None:
            result = Path(directory)
            if not result.is_dir():
                raise FileNotFoundError(f"experiment directory does not exist: {result}")
            return result
        candidates = []
        for root in (
            Path("results/evaluation"),
            Path("Cross Species Pipeline/results/evaluation"),
        ):
            candidates.extend(root.rglob("module_evaluation_*"))
        candidates = [
            candidate for candidate in candidates
            if candidate.is_dir() and list(candidate.glob("batch_*/settings.csv"))
        ]
        if not candidates:
            raise FileNotFoundError(
                "No saved module_evaluation_* experiment was found; pass experiment_dir."
            )
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime)

    experiment_root = find_experiment(experiment_dir)
    if output_dir is None:
        output_root = experiment_root / (
            "pca_pilot_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir()

    def safe_filename(value):
        cleaned = "".join(
            character if character.isalnum() or character in {"-", "_"}
            else "_" for character in str(value)
        ).strip("_")
        return cleaned or "clustering"

    def read_saved_table(run_dir, stem):
        parquet_path = run_dir / f"{stem}.parquet"
        csv_path = run_dir / f"{stem}.csv.gz"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"missing {stem} table in {run_dir}")

    def write_frame(frame, base_path):
        try:
            saved_path = base_path.with_suffix(".parquet")
            frame.to_parquet(saved_path, index=False)
        except (ImportError, ModuleNotFoundError):
            saved_path = base_path.with_suffix(".csv.gz")
            frame.to_csv(saved_path, index=False, compression="gzip")
        return saved_path

    def locate_baseline(name):
        for batch_dir in sorted(experiment_root.glob("batch_*")):
            settings_path = batch_dir / "settings.csv"
            if not settings_path.exists():
                continue
            settings = pd.read_csv(settings_path)
            if "name" not in settings:
                unnamed = [column for column in settings if column.startswith("Unnamed:")]
                if unnamed:
                    settings = settings.rename(columns={unnamed[0]: "name"})
            match = settings[settings["name"].astype(str) == str(name)]
            if match.empty:
                continue
            run_dirs = sorted(
                (batch_dir / "individual_evaluations").glob(
                    f"*_{safe_filename(name)}"
                )
            )
            if len(run_dirs) != 1:
                raise FileNotFoundError(
                    f"expected one evaluation directory for {name}, found {len(run_dirs)}"
                )
            return match.iloc[0], run_dirs[0]
        raise FileNotFoundError(f"saved baseline {name!r} was not found")

    baseline_setting, baseline_dir = locate_baseline(baseline_name)
    graph_parameters = {
        "k_intra": int(baseline_setting["k_intra"]),
        "k_inter": int(baseline_setting["k_inter"]),
        "cutoff_intra": float(baseline_setting["cutoff_intra"]),
        "cutoff_inter": float(baseline_setting["cutoff_inter"]),
    }
    baseline_genes = read_saved_table(baseline_dir, "genes")
    baseline_modules = read_saved_table(baseline_dir, "modules")
    baseline_labels = np.load(baseline_dir / "labels.npy", allow_pickle=False)
    if len(baseline_genes) != n_genes or len(baseline_labels) != n_genes:
        raise ValueError("saved baseline has a different number of genes")

    target_genes = np.asarray(prof.genes).astype(str)
    saved_genes = baseline_genes["gene"].astype(str).to_numpy()
    if np.array_equal(saved_genes, target_genes):
        baseline_order = np.arange(n_genes)
    else:
        index = pd.Index(saved_genes)
        baseline_order = index.get_indexer(target_genes)
        if (baseline_order < 0).any() or index.has_duplicates:
            raise ValueError("saved baseline and prof use different gene universes")
        baseline_genes = baseline_genes.iloc[baseline_order].reset_index(drop=True)
        baseline_labels = baseline_labels[baseline_order]

    species = np.asarray(prof.species).astype(str)
    species_values = [str(value) for value in pd.unique(species)]
    if len(species_values) != 2:
        raise ValueError("the PCA pilot requires exactly two species")
    species_lookup = {value.lower(): value for value in species_values}
    if {"mouse", "human"}.issubset(species_lookup):
        mouse_name = species_lookup["mouse"]
        human_name = species_lookup["human"]
    else:
        mouse_name, human_name = species_values
    split = int(prof.meta["width_a"])
    if not (0 < split < n_genes):
        raise ValueError("prof.meta['width_a'] must split the two species")

    required_intra = max(neighbor_k, graph_parameters["k_intra"])
    required_inter = max(neighbor_k, graph_parameters["k_inter"])
    if reference_neighbor_cache is None:
        print("Building the reference neighbor cache; the no-PCA graph is not reclustered.")
        reference_neighbor_cache = cache_neighbors(
            C_reference, split, max_intra=required_intra,
            max_inter=required_inter, block=matrix_block,
        )
    if (
        reference_neighbor_cache["n"] != n_genes
        or reference_neighbor_cache["intra"][0].shape[1] < required_intra
        or reference_neighbor_cache["inter"][0].shape[1] < required_inter
    ):
        raise ValueError("reference_neighbor_cache is incompatible or too small")

    def finite_quantile(values, q):
        values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        return float(np.quantile(values, q)) if len(values) else np.nan

    def add_baseline_reference_cohesion(modules):
        modules = modules.copy()
        cross_candidates = [
            f"cohesion_{mouse_name}_{human_name}",
            f"cohesion_{human_name}_{mouse_name}",
        ]
        cross_column = next(
            column for column in cross_candidates if column in modules
        )
        modules["reference_cohesion_all"] = modules["cohesion_all"]
        modules["reference_cohesion_mouse_mouse"] = modules[
            f"cohesion_{mouse_name}_{mouse_name}"
        ]
        modules["reference_cohesion_human_human"] = modules[
            f"cohesion_{human_name}_{human_name}"
        ]
        modules["reference_cohesion_cross_species"] = modules[cross_column]
        return modules

    baseline_modules = add_baseline_reference_cohesion(baseline_modules)

    def summarize_clustering(representation, dimension, genes, modules):
        meaningful = modules["n_genes"] >= min_module_size
        meaningful_modules = modules.loc[meaningful]
        meaningful_labels = set(meaningful_modules["module"])
        margin = pd.to_numeric(genes["gene_margin"], errors="coerce")
        finite_margin = margin[np.isfinite(margin)]
        active = genes["degree"].to_numpy() > 0
        cross_connected = active & (genes["cross_degree"].to_numpy() > 0)
        cross_fraction = pd.to_numeric(
            genes["cross_strength_fraction"], errors="coerce"
        ).to_numpy()
        reciprocal_fraction = pd.to_numeric(
            genes["reciprocal_cross_fraction"], errors="coerce"
        ).to_numpy()

        coverage = []
        for species_name in (mouse_name, human_name):
            mask = genes["species"].astype(str).to_numpy() == species_name
            coverage.append(float((genes.loc[mask, "cross_degree"] > 0).mean()))

        def module_median(column):
            if column not in modules:
                return np.nan
            return finite_quantile(meaningful_modules[column], 0.5)

        cross_candidates = [
            f"cohesion_{mouse_name}_{human_name}",
            f"cohesion_{human_name}_{mouse_name}",
        ]
        native_cross = next(
            (column for column in cross_candidates if column in modules), None
        )
        return {
            "representation": representation,
            "pca_dimensions": dimension,
            "n_modules": len(modules),
            f"n_modules_ge_{min_module_size}": int(meaningful.sum()),
            f"fraction_genes_in_modules_ge_{min_module_size}": float(
                genes["module"].isin(meaningful_labels).mean()
            ),
            "singleton_gene_fraction": float((genes["module_size"] == 1).mean()),
            "largest_module_fraction": float(modules["n_genes"].max() / len(genes)),
            "median_gene_margin": finite_quantile(finite_margin, 0.5),
            "p10_gene_margin": finite_quantile(finite_margin, 0.1),
            "fraction_negative_margin": float((finite_margin < 0).mean())
                if len(finite_margin) else np.nan,
            "min_species_cross_coverage": min(coverage),
            "cross_coverage_imbalance": abs(coverage[0] - coverage[1]),
            "median_cross_strength_fraction_cross_connected": finite_quantile(
                cross_fraction[cross_connected], 0.5
            ),
            "reciprocal_gene_coverage": float(
                (genes["reciprocal_cross_degree"] > 0).mean()
            ),
            "median_reciprocal_fraction_cross_connected": finite_quantile(
                reciprocal_fraction[cross_connected], 0.5
            ),
            "median_native_cohesion_mouse_mouse": module_median(
                f"cohesion_{mouse_name}_{mouse_name}"
            ),
            "median_native_cohesion_human_human": module_median(
                f"cohesion_{human_name}_{human_name}"
            ),
            "median_native_cohesion_cross_species": module_median(native_cross)
                if native_cross is not None else np.nan,
            "median_reference_cohesion_mouse_mouse": module_median(
                "reference_cohesion_mouse_mouse"
            ),
            "median_reference_cohesion_human_human": module_median(
                "reference_cohesion_human_human"
            ),
            "median_reference_cohesion_cross_species": module_median(
                "reference_cohesion_cross_species"
            ),
            "median_diversity_mouse": module_median(f"diversity_{mouse_name}"),
            "median_diversity_human": module_median(f"diversity_{human_name}"),
            "median_conserved_diversity": module_median("conserved_diversity"),
        }

    def summarize_graph(representation, dimension, genes, cache):
        degree = genes["degree"].to_numpy(dtype=float)
        cross_degree = genes["cross_degree"].to_numpy(dtype=float)
        within_degree = degree - cross_degree
        result = {
            "representation": representation,
            "pca_dimensions": dimension,
            "n_edges": int(round(degree.sum() / 2)),
            "n_within_edges": int(round(within_degree.sum() / 2)),
            "n_cross_edges": int(round(cross_degree.sum() / 2)),
            "isolated_fraction": float((degree == 0).mean()),
            "median_within_degree": float(np.median(within_degree)),
            "median_cross_degree": float(np.median(cross_degree)),
        }
        species_array = genes["species"].astype(str).to_numpy()
        for species_name in (mouse_name, human_name):
            mask = species_array == species_name
            result[f"cross_edge_coverage_{species_name}"] = float(
                (cross_degree[mask] > 0).mean()
            )
        for kind, k_name, cutoff_name in (
            ("intra", "k_intra", "cutoff_intra"),
            ("inter", "k_inter", "cutoff_inter"),
        ):
            k = graph_parameters[k_name]
            cutoff = graph_parameters[cutoff_name]
            kth_weight = cache[kind][1][:, k - 1]
            result[f"fraction_{kind}_k_binding"] = float(
                (np.isfinite(kth_weight) & (kth_weight >= cutoff)).mean()
            )
        return result

    def row_neighbor_recall(reference_indices, candidate_indices, k):
        reference_indices = reference_indices[:, :k]
        candidate_indices = candidate_indices[:, :k]
        recalls = np.empty(len(reference_indices), dtype=np.float32)
        for start in range(0, len(reference_indices), matrix_block):
            stop = min(start + matrix_block, len(reference_indices))
            reference_block = reference_indices[start:stop]
            candidate_block = candidate_indices[start:stop]
            matches = (
                candidate_block[:, :, None] == reference_block[:, None, :]
            ).any(axis=2)
            valid = candidate_block >= 0
            recalls[start:stop] = (matches & valid).sum(axis=1) / k
        return recalls

    rng = np.random.default_rng(random_state)

    def sample_same_block(start, stop, count):
        left = rng.integers(start, stop, size=count, dtype=np.int64)
        right = rng.integers(start, stop, size=count, dtype=np.int64)
        equal = left == right
        while equal.any():
            right[equal] = rng.integers(start, stop, size=int(equal.sum()))
            equal = left == right
        return left, right

    sampled_pairs = {
        f"{mouse_name}-{mouse_name}": sample_same_block(
            0, split, sampled_pairs_per_type
        ),
        f"{human_name}-{human_name}": sample_same_block(
            split, n_genes, sampled_pairs_per_type
        ),
        f"{mouse_name}-{human_name}": (
            rng.integers(0, split, size=sampled_pairs_per_type, dtype=np.int64),
            rng.integers(
                split, n_genes, size=sampled_pairs_per_type, dtype=np.int64
            ),
        ),
    }
    reference_samples = {
        pair_type: C_reference[left, right].astype(np.float64, copy=False)
        for pair_type, (left, right) in sampled_pairs.items()
    }

    similarity_rows = []
    for pair_type, values in reference_samples.items():
        similarity_rows.append({
            "representation": "no_pca",
            "pca_dimensions": 0,
            "pair_type": pair_type,
            "pearson_with_full_C": 1.0,
            "spearman_with_full_C": 1.0,
            "mean_absolute_error": 0.0,
            "mean_similarity": float(np.mean(values)),
            "sd_similarity": float(np.std(values)),
            "p10_similarity": float(np.quantile(values, 0.1)),
            "median_similarity": float(np.median(values)),
            "p90_similarity": float(np.quantile(values, 0.9)),
        })

    neighbor_rows = []
    for kind in ("intra", "inter"):
        for species_name in (mouse_name, human_name, "overall"):
            neighbor_rows.append({
                "representation": "no_pca",
                "pca_dimensions": 0,
                "neighbor_type": kind,
                "species": species_name,
                "mean_recall": 1.0,
                "median_recall": 1.0,
                "p10_recall": 1.0,
            })

    clustering_rows = [
        summarize_clustering(
            "no_pca", 0, baseline_genes, baseline_modules
        )
    ]
    graph_rows = [
        summarize_graph(
            "no_pca", 0, baseline_genes, reference_neighbor_cache
        )
    ]
    module_sizes = {
        "no_pca": np.sort(baseline_modules["n_genes"].to_numpy())[::-1]
    }
    labels_by_representation = {"no_pca": np.asarray(baseline_labels)}
    active_by_representation = {
        "no_pca": baseline_genes["degree"].to_numpy() > 0
    }
    meaningful_by_representation = {
        "no_pca": baseline_genes["module_size"].to_numpy() >= min_module_size
    }

    print(
        f"Fitting one nested PCA: {n_genes:,} genes x "
        f"{profile_feature_count:,} features -> {max_components} components"
    )
    fit_start = time.perf_counter()
    # StitchedProfiles.features already returns a fresh float32 array, so it
    # can be standardized in place without making a second full-size copy.
    X = prof.features.astype(np.float32, copy=False)
    row_means = X.mean(axis=1, keepdims=True)
    X -= row_means
    row_sd = X.std(axis=1, keepdims=True)
    X /= np.where(row_sd > 0, row_sd, 1).astype(np.float32)
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        whiten=False,
        random_state=random_state,
    )
    pca_scores = pca.fit_transform(X).astype(np.float32, copy=False)
    fit_seconds = time.perf_counter() - fit_start
    del X, row_means, row_sd
    gc.collect()

    np.save(output_root / f"pca_scores_{max_components}.npy", pca_scores)
    np.save(
        output_root / f"pca_components_{max_components}.npy",
        pca.components_.astype(np.float32),
    )
    np.save(output_root / "pca_feature_mean.npy", pca.mean_.astype(np.float32))
    np.save(
        output_root / "pca_explained_variance_ratio.npy",
        pca.explained_variance_ratio_,
    )
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    representation_rows = []
    timing_rows = []

    for dimension in dimensions:
        dimension_start = time.perf_counter()
        representation = f"pca_{dimension:03d}"
        run_dir = output_root / representation
        run_dir.mkdir()
        print(
            f"[{representation}] explained variance "
            f"{cumulative_variance[dimension - 1]:.1%}: building similarity matrix"
        )

        embedding = pca_pearson_embedding(
            pca_scores, dimension, dtype=np.float32
        )

        for pair_type, (left, right) in sampled_pairs.items():
            pca_values = np.einsum(
                "ij,ij->i", embedding[left], embedding[right]
            ).astype(np.float64, copy=False)
            reference_values = reference_samples[pair_type]
            similarity_rows.append({
                "representation": representation,
                "pca_dimensions": dimension,
                "pair_type": pair_type,
                "pearson_with_full_C": float(
                    pearsonr(reference_values, pca_values).statistic
                ),
                "spearman_with_full_C": float(
                    spearmanr(reference_values, pca_values).statistic
                ),
                "mean_absolute_error": float(
                    np.mean(np.abs(reference_values - pca_values))
                ),
                "mean_similarity": float(np.mean(pca_values)),
                "sd_similarity": float(np.std(pca_values)),
                "p10_similarity": float(np.quantile(pca_values, 0.1)),
                "median_similarity": float(np.median(pca_values)),
                "p90_similarity": float(np.quantile(pca_values, 0.9)),
            })

        C_pca, _ = similarity_from_unit_rows(
            embedding,
            return_matrix=True,
            collect_edges=False,
            dtype=np.float32,
            block=matrix_block,
        )
        del embedding

        pca_cache = cache_neighbors(
            C_pca, split, max_intra=required_intra,
            max_inter=required_inter, block=matrix_block,
        )
        for kind in ("intra", "inter"):
            recalls = row_neighbor_recall(
                reference_neighbor_cache[kind][0], pca_cache[kind][0], neighbor_k
            )
            for species_name in (mouse_name, human_name, "overall"):
                mask = (
                    np.ones(n_genes, dtype=bool)
                    if species_name == "overall" else species == species_name
                )
                values = recalls[mask]
                neighbor_rows.append({
                    "representation": representation,
                    "pca_dimensions": dimension,
                    "neighbor_type": kind,
                    "species": species_name,
                    "mean_recall": float(np.mean(values)),
                    "median_recall": float(np.median(values)),
                    "p10_recall": float(np.quantile(values, 0.1)),
                })

        graph = cached_graph(
            pca_cache, prof.genes, prof.species, **graph_parameters
        )
        labels = graph.leiden(
            resolution=resolution, seed=leiden_seed, verbose=False
        )
        evaluation = evaluate_joint_modules(
            labels,
            C_pca,
            graph,
            prof,
            expression_by_species=expression_by_species,
            k_inter=graph_parameters["k_inter"],
            cutoff_inter=graph_parameters["cutoff_inter"],
            min_module_size=2,
            diversity_min_genes=2,
            make_plots=False,
        )
        reference_cohesion = reference_cohesion_table(
            labels, C_reference, species, min_module_size=1
        )
        evaluation["modules"] = evaluation["modules"].merge(
            reference_cohesion, on="module", how="left", validate="one_to_one"
        )

        np.save(run_dir / "labels.npy", labels)
        sparse.save_npz(run_dir / "graph_adjacency.npz", graph.adj)
        write_frame(evaluation["genes"], run_dir / "genes")
        write_frame(evaluation["modules"], run_dir / "modules")
        evaluation["integration"].to_csv(run_dir / "integration.csv", index=False)
        evaluation["summary"].to_frame("value").to_csv(run_dir / "summary.csv")

        graph_rows.append(
            summarize_graph(
                representation, dimension, evaluation["genes"], pca_cache
            )
        )
        clustering_rows.append(
            summarize_clustering(
                representation, dimension,
                evaluation["genes"], evaluation["modules"],
            )
        )
        module_sizes[representation] = np.sort(
            evaluation["modules"]["n_genes"].to_numpy()
        )[::-1]
        labels_by_representation[representation] = np.asarray(labels)
        active_by_representation[representation] = (
            evaluation["genes"]["degree"].to_numpy() > 0
        )
        meaningful_by_representation[representation] = (
            evaluation["genes"]["module_size"].to_numpy() >= min_module_size
        )
        representation_rows.append({
            "representation": representation,
            "pca_dimensions": dimension,
            "cumulative_explained_variance": float(
                cumulative_variance[dimension - 1]
            ),
        })
        elapsed = time.perf_counter() - dimension_start
        timing_rows.append({
            "representation": representation,
            "pca_dimensions": dimension,
            "elapsed_seconds": elapsed,
        })
        print(f"[{representation}] saved in {elapsed / 60:.1f} minutes -> {run_dir}")

        del C_pca, pca_cache, graph, labels, evaluation, reference_cohesion
        gc.collect()

    representation_summary = pd.DataFrame(representation_rows)
    sampled_similarity = pd.DataFrame(similarity_rows)
    neighbor_recall = pd.DataFrame(neighbor_rows)
    graph_summary = pd.DataFrame(graph_rows)
    clustering_summary = pd.DataFrame(clustering_rows)
    timing = pd.DataFrame(timing_rows)

    agreement_rows = []
    representation_order = ["no_pca"] + [f"pca_{d:03d}" for d in dimensions]
    successive_pairs = set(zip(representation_order[:-1], representation_order[1:]))
    for first, second in combinations(representation_order, 2):
        first_labels = labels_by_representation[first]
        second_labels = labels_by_representation[second]
        active_mask = (
            active_by_representation[first] & active_by_representation[second]
        )
        meaningful_mask = (
            meaningful_by_representation[first]
            & meaningful_by_representation[second]
        )

        def masked_ari(mask):
            if int(mask.sum()) < 2:
                return np.nan
            return float(adjusted_rand_score(
                first_labels[mask], second_labels[mask]
            ))

        agreement_rows.append({
            "representation_a": first,
            "representation_b": second,
            "successive_pair": (first, second) in successive_pairs,
            "n_all": n_genes,
            "n_active_both": int(active_mask.sum()),
            "n_meaningful_both": int(meaningful_mask.sum()),
            "ari_all": float(adjusted_rand_score(first_labels, second_labels)),
            "ari_active_both": masked_ari(active_mask),
            "ari_meaningful_both": masked_ari(meaningful_mask),
        })
    partition_agreement = pd.DataFrame(agreement_rows)

    representation_summary.to_csv(
        output_root / "representation_summary.csv", index=False
    )
    sampled_similarity.to_csv(
        output_root / "sampled_similarity.csv", index=False
    )
    neighbor_recall.to_csv(output_root / "neighbor_recall.csv", index=False)
    graph_summary.to_csv(output_root / "graph_summary.csv", index=False)
    clustering_summary.to_csv(
        output_root / "clustering_summary.csv", index=False
    )
    partition_agreement.to_csv(
        output_root / "partition_agreement.csv", index=False
    )
    timing.to_csv(output_root / "timing.csv", index=False)

    config = {
        "experiment_dir": str(experiment_root),
        "baseline_name": baseline_name,
        "baseline_directory": str(baseline_dir),
        "dimensions": list(dimensions),
        "graph_parameters": graph_parameters,
        "resolution": resolution,
        "leiden_seed": leiden_seed,
        "neighbor_k": neighbor_k,
        "sampled_pairs_per_type": sampled_pairs_per_type,
        "random_state": random_state,
        "min_module_size": min_module_size,
        "pca_fit_seconds": fit_seconds,
        "pca_input_dtype": "float32",
        "note": "The no-PCA baseline was loaded from disk and not reclustered.",
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2))

    figures = {}
    pca_dimensions = np.asarray(dimensions)

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.2), dpi=130, constrained_layout=True
    )
    axes[0, 0].plot(
        representation_summary["pca_dimensions"],
        representation_summary["cumulative_explained_variance"],
        marker="o", color="#3274A1",
    )
    axes[0, 0].set(
        xlabel="PCA dimensions", ylabel="Cumulative explained variance",
        title="PCA variance retained",
    )

    pca_similarity = sampled_similarity[
        sampled_similarity["pca_dimensions"] > 0
    ]
    for pair_type, group in pca_similarity.groupby("pair_type", sort=False):
        axes[0, 1].plot(
            group["pca_dimensions"], group["spearman_with_full_C"],
            marker="o", label=pair_type,
        )
    axes[0, 1].set(
        xlabel="PCA dimensions", ylabel="Spearman correlation with full C",
        title="Sampled similarity preservation",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    overall_recall = neighbor_recall[
        (neighbor_recall["pca_dimensions"] > 0)
        & (neighbor_recall["species"] == "overall")
    ]
    for neighbor_type, group in overall_recall.groupby("neighbor_type", sort=False):
        axes[1, 0].plot(
            group["pca_dimensions"], group["mean_recall"],
            marker="o", label=neighbor_type,
        )
    axes[1, 0].set(
        xlabel="PCA dimensions", ylabel=f"Mean neighbor recall@{neighbor_k}",
        title="Neighbor preservation",
    )
    axes[1, 0].legend(frameon=False)

    baseline_graph = graph_summary.iloc[0]
    pca_graph = graph_summary[graph_summary["pca_dimensions"] > 0]
    axes[1, 1].axhline(1, color="0.45", ls="--", lw=1)
    axes[1, 1].plot(
        pca_graph["pca_dimensions"],
        pca_graph["n_within_edges"] / baseline_graph["n_within_edges"],
        marker="o", label="within-edge ratio",
    )
    axes[1, 1].plot(
        pca_graph["pca_dimensions"],
        pca_graph["n_cross_edges"] / baseline_graph["n_cross_edges"],
        marker="o", label="cross-edge ratio",
    )
    axes[1, 1].set(
        xlabel="PCA dimensions", ylabel="Ratio to saved no-PCA graph",
        title="Graph-density change at unchanged cutoffs",
    )
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xticks(pca_dimensions)
    fig.suptitle(f"PCA representation screen against {baseline_name}")
    figures["representation_screen"] = fig

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.2), dpi=130, constrained_layout=True
    )
    versus_baseline = partition_agreement[
        partition_agreement["representation_a"] == "no_pca"
    ].copy()
    versus_baseline["pca_dimensions"] = versus_baseline[
        "representation_b"
    ].str.extract(r"(\d+)$", expand=False).astype(int)
    for column, label in (
        ("ari_all", "all genes"),
        ("ari_active_both", "active in both"),
        ("ari_meaningful_both", "meaningful in both"),
    ):
        axes[0, 0].plot(
            versus_baseline["pca_dimensions"], versus_baseline[column],
            marker="o", label=label,
        )
    axes[0, 0].set(
        xlabel="PCA dimensions", ylabel="ARI with saved no-PCA run_049",
        title="Partition preservation", ylim=(-0.02, 1.02),
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    x_labels = ["no PCA"] + [str(value) for value in dimensions]
    x = np.arange(len(x_labels))
    for column, label in (
        ("median_gene_margin", "median margin"),
        ("p10_gene_margin", "10th-percentile margin"),
    ):
        axes[0, 1].plot(
            x, clustering_summary[column], marker="o", label=label
        )
    axes[0, 1].set(title="Gene separation", ylabel="Gene margin")
    axes[0, 1].legend(frameon=False)

    for column, label in (
        ("min_species_cross_coverage", "minimum species coverage"),
        ("reciprocal_gene_coverage", "reciprocal gene coverage"),
        ("median_cross_strength_fraction_cross_connected", "conditional cross strength"),
    ):
        axes[1, 0].plot(
            x, clustering_summary[column], marker="o", label=label
        )
    axes[1, 0].set(title="Cross-species integration", ylabel="Fraction")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for column, label in (
        ("median_reference_cohesion_mouse_mouse", "mouse cohesion"),
        ("median_reference_cohesion_human_human", "human cohesion"),
        ("median_reference_cohesion_cross_species", "cross cohesion"),
        ("median_conserved_diversity", "conserved diversity"),
    ):
        axes[1, 1].plot(
            x, clustering_summary[column], marker="o", label=label
        )
    axes[1, 1].set(
        title="Quality on common full-profile reference", ylabel="Metric value"
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set_xticks(x, x_labels)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("PCA clustering comparison")
    figures["clustering_comparison"] = fig

    fig, axis = plt.subplots(figsize=(7.8, 5.2), dpi=130, constrained_layout=True)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, len(representation_order))
    )
    for representation, color in zip(representation_order, colors):
        sizes = module_sizes[representation]
        axis.plot(
            np.arange(1, len(sizes) + 1), sizes,
            color=color, lw=1.7, label=representation,
        )
    axis.axhline(
        min_module_size, color="0.45", ls="--", lw=1,
        label=f"minimum size = {min_module_size}",
    )
    axis.set_yscale("log")
    axis.set(
        xlabel="Module rank", ylabel="Genes in module (log scale)",
        title="Ranked module-size distributions",
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.15)
    figures["ranked_module_sizes"] = fig

    plot_paths = {}
    for name, figure in figures.items():
        plot_path = plots_dir / f"{name}.png"
        figure.savefig(plot_path, dpi=250, bbox_inches="tight")
        plot_paths[name] = str(plot_path)
    if show_plots:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)

    total_seconds = fit_seconds + timing["elapsed_seconds"].sum()
    print(
        f"PCA pilot complete in {total_seconds / 60:.1f} minutes -> {output_root}"
    )
    return {
        "representation_summary": representation_summary,
        "sampled_similarity": sampled_similarity,
        "neighbor_recall": neighbor_recall,
        "graph_summary": graph_summary,
        "clustering_summary": clustering_summary,
        "partition_agreement": partition_agreement,
        "timing": timing,
        "figures": figures,
        "plot_paths": plot_paths,
        "output_dir": str(output_root),
    }



def locate_latest_pca_pilot(pilot_dir=None):
    """Locate a completed PCA pilot containing saved nested PCA scores."""
    from pathlib import Path

    if pilot_dir is not None:
        path = Path(pilot_dir)
        if not path.exists():
            raise FileNotFoundError(f"PCA pilot directory does not exist: {path}")
        return path
    roots = [
        Path("results/pca"),
        Path("Cross Species Pipeline/results/pca"),
    ]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(
                path for path in root.rglob("pca_pilot_*")
                if path.is_dir() and list(path.glob("pca_scores_*.npy"))
            )
    if not candidates:
        raise FileNotFoundError(
            "No completed pca_pilot_* directory was found. Pass pilot_dir explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_nested_pca_scores(pilot_dir=None, minimum_dimensions=100):
    """Load the widest saved PCA score matrix from a completed pilot."""
    import re
    import numpy as np

    pilot_dir = locate_latest_pca_pilot(pilot_dir)
    candidates = []
    for path in pilot_dir.glob("pca_scores_*.npy"):
        match = re.search(r"(\d+)$", path.stem)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No pca_scores_*.npy file in {pilot_dir}")
    saved_dimensions, score_path = max(candidates)
    if saved_dimensions < minimum_dimensions:
        raise ValueError(
            f"Saved PCA scores have {saved_dimensions} dimensions; "
            f"at least {minimum_dimensions} are required."
        )
    return np.load(score_path, mmap_mode="r"), pilot_dir, score_path


def run_similarity_calibration(
    prof,
    C_reference,
    reference_edges,
    *,
    pca_scores,
    dimensions=(50, 100),
    output_dir="results/pca/similarity_calibration",
    matrix_block=512,
    sampled_pairs_per_type=100_000,
    random_state=0,
    show_plots=True,
):
    """Compare no-PCA and PCA similarities using the shared EdgeSets workflow.

    The original ``EdgeSets`` object is reused directly. PCA-score rows are
    centered and unit-normalized so their dot products are Pearson correlations.
    PCA distributions then use ``similarity_from_unit_rows`` with the same bin
    edges, leaving one implementation of pair selection and histogram counting.
    """
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Calibration directory is not empty: {output_root}. "
            "Choose a new directory to preserve the existing results."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir()

    dimensions = tuple(sorted({int(value) for value in dimensions}))
    n = len(prof.genes)
    split = int(prof.meta["width_a"])
    if C_reference.shape != (n, n) or pca_scores.shape[0] != n:
        raise ValueError("C_reference, pca_scores, and prof must have the same genes")
    if not dimensions or dimensions[-1] > pca_scores.shape[1]:
        raise ValueError("Requested PCA dimensions are unavailable")
    if reference_edges.meta.get("n") != n or reference_edges.meta.get("width_a") != split:
        raise ValueError("reference_edges does not match prof")

    name_a = str(reference_edges.meta["name_a"])
    name_b = str(reference_edges.meta["name_b"])
    pair_types = (
        f"{name_a}-{name_a}",
        f"{name_b}-{name_b}",
        f"{name_a}-{name_b}",
    )
    representations = ["no_pca", *[f"pca_{value:03d}" for value in dimensions]]
    embeddings = {
        f"pca_{dimension:03d}": pca_pearson_embedding(
            pca_scores, dimension, dtype=np.float32
        )
        for dimension in dimensions
    }
    edge_sets_by_representation = {"no_pca": reference_edges}
    for representation, embedding in embeddings.items():
        print(f"[{representation}] summarizing pair distributions", flush=True)
        _, edge_sets = similarity_from_unit_rows(
            embedding,
            return_matrix=False,
            collect_edges=True,
            split=split,
            name_a=name_a,
            name_b=name_b,
            bins=reference_edges.bins,
            block=matrix_block,
            meta={
                "use_pca": True,
                "n_components": embedding.shape[1],
                "original_features": prof.features.shape[1],
                "similarity_name": "Pearson correlation between PCA-score vectors",
            },
        )
        edge_sets_by_representation[representation] = edge_sets

    histogram_rows, summary_rows = [], []
    centers = (reference_edges.bins[:-1] + reference_edges.bins[1:]) / 2
    for representation, edge_sets in edge_sets_by_representation.items():
        for row in edge_sets.stats.to_dict("records"):
            summary_rows.append({"representation": representation, **row})
        for pair_type in pair_types:
            histogram = edge_sets.counts[pair_type]
            tail = np.cumsum(histogram[::-1])[::-1]
            total = max(histogram.sum(), 1)
            for center, count, tail_count in zip(centers, histogram, tail):
                histogram_rows.append({
                    "representation": representation,
                    "pair_type": pair_type,
                    "bin_center": center,
                    "count": int(count),
                    "density": count / total,
                    "fraction_at_or_above": tail_count / total,
                })
    histograms = pd.DataFrame(histogram_rows)
    distribution_summary = pd.DataFrame(summary_rows)
    histograms.to_csv(
        output_root / "similarity_histograms.csv.gz",
        index=False, compression="gzip",
    )
    distribution_summary.to_csv(
        output_root / "similarity_distribution_summary.csv", index=False
    )

    rng = np.random.default_rng(random_state)

    def same_species_sample(start, stop, size):
        left = rng.integers(start, stop, size=size, dtype=np.int32)
        right = rng.integers(start, stop, size=size, dtype=np.int32)
        equal = left == right
        while equal.any():
            right[equal] = rng.integers(start, stop, size=int(equal.sum()))
            equal = left == right
        return left, right

    sampled_indices = {
        pair_types[0]: same_species_sample(0, split, sampled_pairs_per_type),
        pair_types[1]: same_species_sample(split, n, sampled_pairs_per_type),
        pair_types[2]: (
            rng.integers(0, split, size=sampled_pairs_per_type, dtype=np.int32),
            rng.integers(split, n, size=sampled_pairs_per_type, dtype=np.int32),
        ),
    }
    sampled, fidelity_rows = {}, []
    for pair_type, (left, right) in sampled_indices.items():
        original = np.asarray(C_reference[left, right], dtype=np.float32)
        sampled[(pair_type, "no_pca")] = original
        for representation, embedding in embeddings.items():
            reduced = np.einsum(
                "ij,ij->i", embedding[left], embedding[right]
            ).astype(np.float32)
            sampled[(pair_type, representation)] = reduced
            residual = reduced - original
            fidelity_rows.append({
                "representation": representation,
                "pair_type": pair_type,
                "pearson_with_no_pca": pearsonr(original, reduced).statistic,
                "spearman_with_no_pca": spearmanr(original, reduced).statistic,
                "mean_absolute_error": float(np.mean(np.abs(residual))),
                "mean_residual": float(np.mean(residual)),
            })
    fidelity = pd.DataFrame(fidelity_rows)
    fidelity.to_csv(output_root / "sampled_similarity_fidelity.csv", index=False)
    archive = {}
    for pair_type, (left, right) in sampled_indices.items():
        tag = pair_type.replace("-", "_")
        archive[f"{tag}_left"] = left
        archive[f"{tag}_right"] = right
        for representation in representations:
            archive[f"{tag}_{representation}"] = sampled[(pair_type, representation)]
    np.savez_compressed(output_root / "sampled_pair_similarities.npz", **archive)

    figures = {}
    # Reuse the original distribution plot for every representation.
    for representation, edge_sets in edge_sets_by_representation.items():
        figure = plot_edges(edge_sets, log=True, cumulative=False)
        figure.suptitle(f"Pairwise similarities: {representation.replace('_', ' ').upper()}")
        figures[f"pair_distributions_{representation}"] = figure

    figure, axes = plt.subplots(
        1, 3, figsize=ANALYSIS_WIDE_FIGSIZE,
        dpi=ANALYSIS_FIGURE_DPI, constrained_layout=True,
        sharex=True,
    )
    for axis, pair_type in zip(axes, pair_types):
        for representation in representations:
            edge_sets = edge_sets_by_representation[representation]
            axis.plot(
                centers, edge_sets.density(pair_type), lw=1.7,
                color=REPRESENTATION_COLORS[representation],
                label=representation.replace("_", " ").upper(),
            )
        axis.set_yscale("log")
        axis.set(
            title=pair_type.replace("-", "–").title(),
            xlabel="Second-order similarity", ylabel="Pair fraction",
        )
    axes[0].legend(frameon=False)
    style_analysis_axes(axes)
    figure.suptitle("Second-order similarity distributions")
    figures["similarity_distributions_comparison"] = figure

    figure, axes = plt.subplots(
        1, 3, figsize=ANALYSIS_WIDE_FIGSIZE,
        dpi=ANALYSIS_FIGURE_DPI, constrained_layout=True,
        sharex=True, sharey=True,
    )
    for axis, pair_type in zip(axes, pair_types):
        for representation in representations:
            edge_sets = edge_sets_by_representation[representation]
            histogram = edge_sets.counts[pair_type]
            survival = np.cumsum(histogram[::-1])[::-1] / histogram.sum()
            mask = centers >= 0.3
            axis.plot(
                centers[mask], survival[mask], lw=1.7,
                color=REPRESENTATION_COLORS[representation],
                label=representation.replace("_", " ").upper(),
            )
        axis.set_yscale("log")
        axis.set(
            xlim=(0.3, 1.0),
            title=pair_type.replace("-", "–").title(),
            xlabel="Similarity cutoff",
            ylabel="Fraction at or above cutoff",
        )
    axes[0].legend(frameon=False)
    style_analysis_axes(axes)
    figure.suptitle("Upper tails used to calibrate similarity cutoffs")
    figures["similarity_upper_tail_ccdf"] = figure

    for representation in representations[1:]:
        display_name = representation.replace("_", " ").upper()
        figure, axes = plt.subplots(
            1, 3, figsize=ANALYSIS_WIDE_FIGSIZE,
            dpi=ANALYSIS_FIGURE_DPI, constrained_layout=True,
        )
        for axis, pair_type in zip(axes, pair_types):
            original = sampled[(pair_type, "no_pca")]
            reduced = sampled[(pair_type, representation)]
            axis.hexbin(
                original, reduced, gridsize=70, bins="log",
                mincnt=1, cmap="viridis",
            )
            limits = (
                min(original.min(), reduced.min()),
                max(original.max(), reduced.max()),
            )
            axis.plot(limits, limits, ls="--", lw=1, color="white")
            axis.set(
                title=pair_type.replace("-", "–").title(),
                xlabel="No-PCA Pearson similarity",
                ylabel=f"{display_name} Pearson similarity",
            )
        style_analysis_axes(axes)
        figure.suptitle(f"No-PCA versus {display_name} pair similarities")
        figures[f"no_pca_vs_{representation}"] = figure

        figure, axes = plt.subplots(
            1, 3, figsize=ANALYSIS_WIDE_FIGSIZE,
            dpi=ANALYSIS_FIGURE_DPI, constrained_layout=True,
            sharex=True, sharey=True,
        )
        residual_rows = []
        residual_edges = np.linspace(-1, 1, 41)
        residual_centers = (residual_edges[:-1] + residual_edges[1:]) / 2
        for axis, pair_type in zip(axes, pair_types):
            original = sampled[(pair_type, "no_pca")]
            residual = sampled[(pair_type, representation)] - original
            groups = np.digitize(original, residual_edges) - 1
            median, low, high = [], [], []
            for index, center in enumerate(residual_centers):
                values = residual[groups == index]
                median.append(np.nan if not len(values) else np.median(values))
                low.append(np.nan if not len(values) else np.quantile(values, 0.10))
                high.append(np.nan if not len(values) else np.quantile(values, 0.90))
                residual_rows.append({
                    "representation": representation,
                    "pair_type": pair_type,
                    "original_similarity_bin": center,
                    "n_pairs": len(values),
                    "median_residual": median[-1],
                    "p10_residual": low[-1],
                    "p90_residual": high[-1],
                })
            color = REPRESENTATION_COLORS[representation]
            axis.plot(residual_centers, median, color=color, lw=1.8)
            axis.fill_between(residual_centers, low, high, color=color, alpha=0.18)
            axis.axhline(0, color="0.4", ls="--", lw=1)
            axis.set(
                title=pair_type.replace("-", "–").title(),
                xlabel="No-PCA similarity",
                ylabel="PCA minus no-PCA similarity",
            )
        style_analysis_axes(axes)
        pd.DataFrame(residual_rows).to_csv(
            output_root / f"{representation}_residual_by_original_bin.csv",
            index=False,
        )
        figure.suptitle(f"Similarity changes introduced by {display_name}")
        figures[f"{representation}_residuals"] = figure

    plot_paths = save_analysis_figures(figures, plots_dir)
    if show_plots:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)

    (output_root / "configuration.json").write_text(json.dumps({
        "representations": representations,
        "dimensions": dimensions,
        "bins_reused_from_reference_edges": len(reference_edges.bins) - 1,
        "sampled_pairs_per_type": sampled_pairs_per_type,
        "random_state": random_state,
        "plot_format": ANALYSIS_PLOT_FORMAT,
        "plot_dpi": ANALYSIS_SAVE_DPI,
        "similarity_note": (
            "no_pca uses Pearson correlation across ortholog features; PCA uses "
            "Pearson correlation across centered, nested PCA-score vectors"
        ),
    }, indent=2))
    print(f"Similarity calibration written to: {output_root}")
    return {
        "edge_sets": edge_sets_by_representation,
        "distribution_summary": distribution_summary,
        "histograms": histograms,
        "fidelity": fidelity,
        "figures": figures,
        "plot_paths": plot_paths,
        "output_dir": str(output_root),
    }



def load_saved_pca_components(pilot_dir=None, minimum_pcs=5):
    """Load the widest saved PCA component matrix from a completed pilot."""
    import re
    import numpy as np

    pilot_dir = locate_latest_pca_pilot(pilot_dir)
    candidates = []
    for path in pilot_dir.glob("pca_components_*.npy"):
        match = re.search(r"(\d+)$", path.stem)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No pca_components_*.npy file in {pilot_dir}")
    saved_pcs, component_path = max(candidates)
    if saved_pcs < minimum_pcs:
        raise ValueError(
            f"Saved PCA components contain {saved_pcs} PCs; "
            f"at least {minimum_pcs} are required."
        )
    return np.load(component_path, mmap_mode="r"), pilot_dir, component_path


def plot_pca_ortholog_interpretation(
    prof,
    res,
    pca_components,
    *,
    expression_by_species,
    pcs=range(1, 6),
    n_bar_per_tail=15,
    n_maps_per_tail=5,
    output_dir="results/pca/component_interpretation_pc1_pc5",
    show_plots=True,
    point_size=None,
):
    """Plot PC loadings and spatial maps for their strongest ortholog anchors.

    A PCA feature is one aligned ortholog pair, so component column ``i`` maps
    to ``res.genes_a[i]`` and ``res.genes_b[i]``. Spatial colors are separately
    scaled to [0, 1] within each gene and species after 1st/99th-percentile
    clipping; they compare patterns, not absolute expression between species.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from scipy import sparse

    components = np.asarray(pca_components)
    pcs = tuple(dict.fromkeys(int(pc) for pc in pcs))
    if not pcs or min(pcs) < 1 or max(pcs) > components.shape[0]:
        raise ValueError("pcs must use one-based indices present in pca_components")
    if n_bar_per_tail < 1 or n_maps_per_tail < 1:
        raise ValueError("n_bar_per_tail and n_maps_per_tail must be positive")

    n_orthologs = int(prof.meta["n_ortho"])
    if components.shape[1] != n_orthologs:
        raise ValueError(
            f"PCA has {components.shape[1]:,} features but prof contains "
            f"{n_orthologs:,} ortholog anchors"
        )
    if len(res.genes_a) < n_orthologs or len(res.genes_b) < n_orthologs:
        raise ValueError("res does not contain every aligned ortholog anchor")

    name_a = str(prof.meta["name_a"])
    name_b = str(prof.meta["name_b"])
    missing_species = {name_a, name_b} - set(expression_by_species)
    if missing_species:
        raise ValueError(f"Missing expression data for {sorted(missing_species)}")

    genes_a = np.asarray(res.genes_a[:n_orthologs], dtype=str)
    genes_b = np.asarray(res.genes_b[:n_orthologs], dtype=str)
    class_keys = (
        np.asarray(res.pairs["class_key"].iloc[:n_orthologs], dtype=str)
        if hasattr(res, "pairs") and "class_key" in res.pairs
        else np.repeat("", n_orthologs)
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    spatial = {}
    for species_name in (name_a, name_b):
        adata = expression_by_species[species_name]
        if "spatial" not in adata.obsm:
            raise KeyError(f"{species_name} AnnData has no obsm['spatial'] coordinates")
        coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError(f"{species_name} spatial coordinates must have two columns")
        exact = {str(gene): str(gene) for gene in adata.var_names}
        upper = {}
        for gene in adata.var_names:
            upper.setdefault(str(gene).upper(), str(gene))
        spatial[species_name] = {
            "adata": adata,
            "coordinates": coordinates[:, :2],
            "exact": exact,
            "upper": upper,
        }

    def expression_vector(species_name, requested_gene):
        info = spatial[species_name]
        matched_gene = info["exact"].get(str(requested_gene))
        if matched_gene is None:
            matched_gene = info["upper"].get(str(requested_gene).upper())
        if matched_gene is None:
            return None, None
        values = info["adata"][:, matched_gene].X
        if sparse.issparse(values):
            values = values.toarray()
        values = np.asarray(values, dtype=float).reshape(-1)
        return values, matched_gene

    def relative_expression(values):
        finite = np.isfinite(values)
        scaled = np.full(values.shape, np.nan, dtype=float)
        if not finite.any():
            return scaled
        low, high = np.nanquantile(values[finite], [0.01, 0.99])
        if not np.isfinite(high) or high <= low:
            low = float(np.nanmin(values[finite]))
            high = float(np.nanmax(values[finite]))
        if high <= low:
            scaled[finite] = 0.0
        else:
            scaled[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
        return scaled

    def draw_spatial(axis, species_name, gene_name):
        values, matched_gene = expression_vector(species_name, gene_name)
        coordinates = spatial[species_name]["coordinates"]
        if values is None:
            axis.text(
                0.5, 0.5, f"{gene_name}\nnot found", ha="center", va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            return None
        scaled = relative_expression(values)
        valid = np.isfinite(scaled) & np.isfinite(coordinates).all(axis=1)
        order = np.argsort(scaled[valid], kind="stable")
        xy = coordinates[valid][order]
        color = scaled[valid][order]
        size = (
            float(point_size)
            if point_size is not None
            else max(0.08, min(2.0, 12_000 / max(len(xy), 1)))
        )
        axis.scatter(
            xy[:, 0], xy[:, 1], c=color, s=size, cmap="viridis",
            vmin=0, vmax=1, linewidths=0, rasterized=True,
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_axis_off()
        return matched_gene

    all_rows = []
    selected_rows = []
    plot_paths = {}

    for pc in pcs:
        loading = np.asarray(components[pc - 1], dtype=float)
        finite = np.isfinite(loading)
        if not finite.all():
            raise ValueError(f"PC{pc} contains non-finite loadings")
        positive_candidates = np.flatnonzero(loading > 0)
        negative_candidates = np.flatnonzero(loading < 0)
        positive = positive_candidates[
            np.argsort(loading[positive_candidates])[::-1]
        ][:n_bar_per_tail]
        negative = negative_candidates[
            np.argsort(loading[negative_candidates])
        ][:n_bar_per_tail]
        map_positive = positive[:min(n_maps_per_tail, len(positive))]
        map_negative = negative[:min(n_maps_per_tail, len(negative))]
        map_indices = np.concatenate([map_positive, map_negative])
        map_poles = (["positive"] * len(map_positive) + ["negative"] * len(map_negative))

        squared = np.square(loading)
        squared_total = squared.sum()
        contribution = squared / squared_total if squared_total > 0 else np.zeros_like(squared)
        ranks = np.empty(n_orthologs, dtype=int)
        ranks[np.argsort(np.abs(loading))[::-1]] = np.arange(1, n_orthologs + 1)
        for index in range(n_orthologs):
            all_rows.append({
                "pc": pc,
                "ortholog_index": index,
                "class_key": class_keys[index],
                f"{name_a}_gene": genes_a[index],
                f"{name_b}_gene": genes_b[index],
                "loading": loading[index],
                "absolute_loading": abs(loading[index]),
                "squared_loading_contribution": contribution[index],
                "absolute_loading_rank": ranks[index],
            })

        pair_labels = np.asarray([
            f"{genes_a[index]} ↔ {genes_b[index]}" for index in range(n_orthologs)
        ])
        bar_indices = np.concatenate([negative, positive[::-1]])
        bar_values = loading[bar_indices]
        bar_labels = pair_labels[bar_indices]
        bar_colors = np.where(bar_values < 0, "#3274A1", "#E1812C")
        fig_height = max(7.0, 0.30 * len(bar_indices))
        figure, axis = plt.subplots(
            figsize=(9.5, fig_height), dpi=ANALYSIS_FIGURE_DPI,
            constrained_layout=True,
        )
        axis.barh(np.arange(len(bar_indices)), bar_values, color=bar_colors)
        axis.set_yticks(np.arange(len(bar_indices)), labels=bar_labels, fontsize=8)
        axis.axvline(0, color="0.35", lw=0.9)
        axis.set(
            xlabel="PCA loading",
            title=(
                f"PC{pc}: {len(negative)} most negative and "
                f"{len(positive)} most positive ortholog anchors"
            ),
        )
        style_analysis_axes([axis])
        bar_path = output_root / f"pc{pc:02d}_loading_bar.{ANALYSIS_PLOT_FORMAT}"
        figure.savefig(bar_path, dpi=ANALYSIS_SAVE_DPI, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(figure)
        plot_paths[f"pc{pc:02d}_loading_bar"] = str(bar_path)

        effective_anchors = (
            float(1.0 / np.square(contribution).sum())
            if np.square(contribution).sum() > 0 else np.nan
        )
        sorted_magnitude = np.sort(np.abs(loading))[::-1]
        cumulative = np.cumsum(np.sort(contribution)[::-1])
        figure, axes = plt.subplots(
            1, 2, figsize=(11.5, 4.2), dpi=ANALYSIS_FIGURE_DPI,
            constrained_layout=True,
        )
        axes[0].hist(loading, bins=80, color="#6A51A3", alpha=0.85)
        axes[0].axvline(0, color="0.35", lw=0.9)
        axes[0].set(
            xlabel="PCA loading", ylabel="Number of ortholog anchors",
            title=f"PC{pc} loading distribution",
        )
        axes[1].plot(
            np.arange(1, n_orthologs + 1), sorted_magnitude,
            color="#6A51A3", lw=1.6,
        )
        axes[1].set_xscale("log")
        axes[1].set(
            xlabel="Absolute-loading rank", ylabel="Absolute loading",
            title=(
                f"Ranked magnitude | effective anchors ≈ {effective_anchors:,.0f}\n"
                f"top {2 * n_maps_per_tail} map anchors: "
                f"{contribution[map_indices].sum():.1%} of squared loading"
            ),
        )
        style_analysis_axes(axes)
        distribution_path = (
            output_root / f"pc{pc:02d}_loading_distribution.{ANALYSIS_PLOT_FORMAT}"
        )
        figure.savefig(
            distribution_path, dpi=ANALYSIS_SAVE_DPI, bbox_inches="tight"
        )
        if show_plots:
            plt.show()
        plt.close(figure)
        plot_paths[f"pc{pc:02d}_loading_distribution"] = str(distribution_path)

        figure, axes = plt.subplots(
            len(map_indices), 2,
            figsize=(10.5, max(15.0, 2.5 * len(map_indices))),
            dpi=ANALYSIS_FIGURE_DPI, constrained_layout=True,
            squeeze=False,
        )
        for row_number, (index, pole) in enumerate(zip(map_indices, map_poles)):
            genes = {name_a: genes_a[index], name_b: genes_b[index]}
            rank_within_pole = row_number + 1 if pole == "positive" else (
                row_number - len(map_positive) + 1
            )
            for column_number, species_name in enumerate((name_a, name_b)):
                axis = axes[row_number, column_number]
                matched = draw_spatial(axis, species_name, genes[species_name])
                shown_gene = matched or genes[species_name]
                axis.set_title(
                    f"{pole.title()} {rank_within_pole} | {species_name}: {shown_gene}\n"
                    f"loading {loading[index]:+.5f}",
                    fontsize=9,
                )
            selected_rows.append({
                "pc": pc,
                "pole": pole,
                "rank_within_pole": rank_within_pole,
                "ortholog_index": int(index),
                "class_key": class_keys[index],
                f"{name_a}_gene": genes_a[index],
                f"{name_b}_gene": genes_b[index],
                "loading": loading[index],
                "squared_loading_contribution": contribution[index],
            })
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
            ax=axes, fraction=0.015, pad=0.01,
        )
        colorbar.set_label("Relative expression within gene and species (1st–99th percentile)")
        figure.suptitle(
            f"PC{pc}: spatial expression of the strongest ortholog anchors",
            fontsize=14,
        )
        spatial_path = output_root / f"pc{pc:02d}_spatial_maps.{ANALYSIS_PLOT_FORMAT}"
        figure.savefig(spatial_path, dpi=ANALYSIS_SAVE_DPI, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(figure)
        plot_paths[f"pc{pc:02d}_spatial_maps"] = str(spatial_path)

    loading_table = pd.DataFrame(all_rows).sort_values(
        ["pc", "absolute_loading_rank"]
    )
    selected_table = pd.DataFrame(selected_rows).sort_values(
        ["pc", "pole", "rank_within_pole"]
    )
    loading_table.to_csv(
        output_root / "pc1_pc5_all_ortholog_loadings.csv.gz",
        index=False, compression="gzip",
    )
    selected_table.to_csv(
        output_root / "pc1_pc5_spatially_plotted_orthologs.csv", index=False
    )
    print(f"PC1–PC5 interpretation written to: {output_root}")
    return {
        "loadings": loading_table,
        "spatially_plotted_orthologs": selected_table,
        "plot_paths": plot_paths,
        "output_dir": str(output_root),
    }



def generate_sobol_settings(
    parameter_ranges,
    *,
    n_settings=32,
    seed=0,
    k_sampling="log",
):
    """Generate well-spread four-parameter settings from a scrambled Sobol sequence."""
    import math
    import numpy as np
    import pandas as pd
    from scipy.stats import qmc

    required = ("k_intra", "k_inter", "cutoff_intra", "cutoff_inter")
    missing = [name for name in required if name not in parameter_ranges]
    if missing:
        raise ValueError(f"Missing parameter ranges: {missing}")
    if n_settings < 2 or n_settings & (n_settings - 1):
        raise ValueError("n_settings must be a power of two, such as 16, 32, or 64")
    if k_sampling not in {"linear", "log"}:
        raise ValueError("k_sampling must be 'linear' or 'log'")

    bounds = {}
    for name in required:
        value = parameter_ranges[name]
        if value is None or len(value) != 2 or any(bound is None for bound in value):
            raise ValueError(
                f"Fill in a numeric (minimum, maximum) range for {name}"
            )
        low, high = map(float, value)
        if not low < high:
            raise ValueError(f"{name} must have minimum < maximum")
        if name.startswith("k_") and low < 1:
            raise ValueError(f"{name} must be positive")
        if name.startswith("cutoff_") and not (0 < low < high <= 1):
            raise ValueError(f"{name} must lie in (0, 1]")
        bounds[name] = (low, high)

    # Oversampling protects against duplicate integer k combinations. The
    # accepted settings remain in Sobol order, and the first 32 points are a
    # balanced power-of-two prefix when no rounding duplicates occur.
    draw_count = max(32, n_settings * 8)
    exponent = int(math.ceil(math.log2(draw_count)))
    unit_points = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(exponent)
    rows, seen = [], set()
    for sobol_index, point in enumerate(unit_points):
        values = {}
        for coordinate, name in enumerate(required):
            low, high = bounds[name]
            if name.startswith("k_"):
                if k_sampling == "log":
                    value = np.exp(np.log(low) + point[coordinate] * (np.log(high) - np.log(low)))
                else:
                    value = low + point[coordinate] * (high - low)
                values[name] = int(np.clip(np.rint(value), np.ceil(low), np.floor(high)))
            else:
                values[name] = float(low + point[coordinate] * (high - low))
        identity = (
            values["k_intra"], values["k_inter"],
            round(values["cutoff_intra"], 10), round(values["cutoff_inter"], 10),
        )
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({"sobol_id": len(rows), "sobol_index": sobol_index, **values})
        if len(rows) == n_settings:
            break
    if len(rows) != n_settings:
        raise RuntimeError("Unable to generate enough unique Sobol settings")
    return pd.DataFrame(rows)


def plot_sobol_search_summaries(
    summary, output_dir, *, species, show_plots=True
):
    """Save compact overviews across all Sobol settings and representations."""
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    plots_dir = output_dir / "summary_plots"
    plots_dir.mkdir(exist_ok=True)
    figures = {}
    species = list(dict.fromkeys(str(value) for value in species))
    cross_cohesion = next(
        (column for column in summary if column.startswith("median_cohesion_")
         and species[0] in column and species[1] in column),
        None,
    )

    panel_specs = [
        (["p10_gene_margin", "median_gene_margin"], "Gene separation", "Margin"),
        (["cross_edge_coverage", "reciprocal_gene_coverage"], "Species integration", "Gene fraction"),
        (["median_cross_strength_fraction", "median_reciprocal_cross_fraction"], "Cross-species support", "Fraction"),
        ([f"median_cohesion_{species[0]}_{species[0]}", f"median_cohesion_{species[1]}_{species[1]}", cross_cohesion], "Native module cohesion", "Similarity"),
        ([f"median_diversity_{species[0]}", f"median_diversity_{species[1]}", "median_conserved_diversity"], "Module diversity", "Diversity"),
        (["n_modules", "n_modules_ge_10"], "Module counts", "Number of modules"),
    ]

    for representation, frame in summary.groupby("representation", sort=False):
        frame = frame.sort_values("sobol_id")
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), dpi=ANALYSIS_FIGURE_DPI, constrained_layout=True)
        for axis, (columns, title, ylabel) in zip(axes.flat, panel_specs):
            for column in columns:
                if column and column in frame:
                    axis.scatter(frame.sobol_id + 1, frame[column], s=24, alpha=0.8, label=column.replace("median_", ""))
            axis.set(title=title, xlabel="Sobol setting", ylabel=ylabel)
            axis.grid(alpha=0.15)
            axis.legend(frameon=False, fontsize=7)
        style_analysis_axes(axes)
        fig.suptitle(
            f"Quasi-random clustering screen: {representation.replace('_', ' ').upper()}"
        )
        figures[f"{representation}_metric_overview"] = fig

    comparison_metrics = [
        "p10_gene_margin", "cross_edge_coverage", "reciprocal_gene_coverage",
        "median_cross_strength_fraction", "median_conserved_diversity",
        "median_reference_cohesion_cross_species",
    ]
    available = [metric for metric in comparison_metrics if metric in summary]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), dpi=ANALYSIS_FIGURE_DPI, constrained_layout=True)
    representation_order = list(dict.fromkeys(summary.representation.tolist()))
    for axis, metric in zip(axes.flat, available):
        values = [
            summary.loc[summary.representation == representation, metric].dropna().to_numpy()
            for representation in representation_order
        ]
        axis.boxplot(
            values,
            tick_labels=[value.replace("_", " ").upper() for value in representation_order],
            showfliers=False,
        )
        axis.set(title=metric.replace("_", " "), ylabel="Value")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.15)
    for axis in axes.flat[len(available):]:
        axis.set_visible(False)
    style_analysis_axes(axes)
    fig.suptitle("Metric distributions across Sobol settings")
    figures["representation_metric_distributions"] = fig
    save_analysis_figures(figures, plots_dir)

    if show_plots:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)
    return figures


def run_sobol_pca_search(
    prof,
    C_reference,
    *,
    pca_scores,
    parameter_ranges_by_representation,
    expression_by_species,
    n_settings=32,
    resolution=1.0,
    leiden_seed=0,
    sobol_seed=0,
    k_sampling="log",
    batch_size=4,
    matrix_block=512,
    output_dir="results/sobol/quasirandom_pca_pcc_32",
    show_summary_plots=True,
):
    """Run resumable Sobol searches for no-PCA, PCA-50, and PCA-100.

    PCA graph weights are Pearson correlations between centered PCA-score
    vectors. Each representation is materialized one at a time. Its largest requested
    neighbor cache is built once, and full module evaluations are written in
    batches so graphs and gene-level results do not accumulate in memory.
    """
    from pathlib import Path
    from datetime import datetime
    import gc
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    representations = ("no_pca", "pca_050", "pca_100")
    missing = set(representations) - set(parameter_ranges_by_representation)
    if missing:
        raise ValueError(f"Missing representation ranges: {sorted(missing)}")
    if n_settings % batch_size:
        raise ValueError("n_settings must be divisible by batch_size")
    if C_reference.shape != (len(prof.genes), len(prof.genes)):
        raise ValueError("C_reference does not match prof")
    if pca_scores.shape[0] != len(prof.genes) or pca_scores.shape[1] < 100:
        raise ValueError("pca_scores must contain all genes and at least 100 PCs")

    designs = {
        representation: generate_sobol_settings(
            parameter_ranges_by_representation[representation],
            n_settings=n_settings, seed=sobol_seed, k_sampling=k_sampling,
        )
        for representation in representations
    }
    configuration = {
        "representations": list(representations),
        "n_settings_per_representation": n_settings,
        "resolution": resolution,
        "leiden_seed": leiden_seed,
        "sobol_seed": sobol_seed,
        "k_sampling": k_sampling,
        "batch_size": batch_size,
        "parameter_ranges": parameter_ranges_by_representation,
        "similarity_note": "no_pca uses Pearson across ortholog features; PCA uses Pearson across centered nested PCA-score vectors",
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "configuration.json"
    serialized = json.loads(json.dumps(configuration))
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != serialized:
            raise ValueError(
                f"Existing search configuration differs in {output_root}. "
                "Use a new output_dir for changed ranges or settings."
            )
        print(f"Resuming compatible search in {output_root}")
    else:
        config_path.write_text(json.dumps(serialized, indent=2))

    all_representation_summaries = []
    split = int(prof.meta["width_a"])

    for representation in representations:
        representation_dir = output_root / representation
        representation_dir.mkdir(exist_ok=True)
        design = designs[representation].copy()
        design.insert(0, "name", [f"{representation}_s{value + 1:02d}" for value in design.sobol_id])
        design.to_csv(representation_dir / "sobol_design.csv", index=False)

        expected_batch_summaries = [
            representation_dir / f"batch_{batch_number:02d}" / "batch_settings_summary.csv"
            for batch_number in range(n_settings // batch_size)
        ]
        if all(path.exists() for path in expected_batch_summaries):
            print(f"[{representation}] all batches are complete; loading saved summaries")
            representation_summary = pd.concat(
                [pd.read_csv(path) for path in expected_batch_summaries],
                ignore_index=True,
            )
            representation_summary["representation"] = representation
            representation_summary = representation_summary.sort_values("sobol_id")
            representation_summary.to_csv(
                representation_dir / "all_settings_summary.csv", index=False
            )
            all_representation_summaries.append(representation_summary)
            continue

        if representation == "no_pca":
            C_current = C_reference
        else:
            dimension = int(representation.split("_")[1])
            print(f"[{representation}] building dense similarity matrix", flush=True)
            embedding = pca_pearson_embedding(
                pca_scores, dimension, dtype=np.float32
            )
            C_current, _ = similarity_from_unit_rows(
                embedding,
                return_matrix=True,
                collect_edges=False,
                dtype=np.float32,
                block=matrix_block,
            )
            del embedding

        max_intra = int(design.k_intra.max())
        max_inter = int(design.k_inter.max())
        print(
            f"[{representation}] caching up to {max_intra} intra and "
            f"{max_inter} inter neighbors per gene",
            flush=True,
        )
        search_cache = cache_neighbors(
            C_current, split, max_intra=max_intra,
            max_inter=max_inter, block=matrix_block,
        )

        batch_summaries = []
        for batch_number, start in enumerate(range(0, n_settings, batch_size)):
            batch_frame = design.iloc[start:start + batch_size].copy()
            batch_dir = representation_dir / f"batch_{batch_number:02d}"
            saved_batch_summary = batch_dir / "batch_settings_summary.csv"
            if saved_batch_summary.exists():
                print(f"[{representation}] skipping completed batch {batch_number + 1}")
                batch_summaries.append(pd.read_csv(saved_batch_summary))
                continue
            if batch_dir.exists() and any(batch_dir.iterdir()):
                raise RuntimeError(
                    f"Incomplete batch directory found: {batch_dir}. "
                    "Delete that batch directory before resuming."
                )

            clustering_batch = []
            for row in batch_frame.itertuples(index=False):
                parameters = {
                    "k_intra": int(row.k_intra),
                    "k_inter": int(row.k_inter),
                    "cutoff_intra": float(row.cutoff_intra),
                    "cutoff_inter": float(row.cutoff_inter),
                }
                graph = cached_graph(
                    search_cache, prof.genes, prof.species, **parameters
                )
                clustering_batch.append({
                    "name": row.name,
                    "representation": representation,
                    "sobol_id": int(row.sobol_id),
                    **parameters,
                    "G": graph,
                    "labels": graph.leiden(
                        resolution=resolution, seed=leiden_seed, verbose=False
                    ),
                })

            batch_result = evaluate_clusterings(
                clustering_batch,
                C_current,
                prof,
                expression_by_species=expression_by_species,
                min_module_size=10,
                include_pairwise=False,
                make_plots=True,
                show_plots=False,
                max_clusterings=max(10, batch_size),
                output_dir=batch_dir,
                plot_format=ANALYSIS_PLOT_FORMAT,
                return_figures=False,
            )
            batch_summary = batch_result["summary"].reset_index()
            first_column = batch_summary.columns[0]
            batch_summary = batch_summary.rename(columns={first_column: "name"})
            settings = batch_result["settings"].reset_index()
            batch_summary = batch_summary.merge(settings, on="name", how="left")

            reference_rows = []
            for specification in clustering_batch:
                reference_table = reference_cohesion_table(
                    specification["labels"], C_reference,
                    np.asarray(prof.species), min_module_size=10,
                )
                reference_rows.append({
                    "name": specification["name"],
                    **{
                        f"median_{column}": float(reference_table[column].median())
                        for column in reference_table
                        if column != "module"
                    },
                })
            batch_summary = batch_summary.merge(
                pd.DataFrame(reference_rows), on="name", how="left"
            )
            batch_summary["batch"] = batch_number
            batch_summary.to_csv(saved_batch_summary, index=False)
            batch_summaries.append(batch_summary)
            print(
                f"[{representation}] completed batch {batch_number + 1}/"
                f"{n_settings // batch_size} -> {batch_dir}",
                flush=True,
            )
            plt.close("all")
            del batch_result, clustering_batch, graph, batch_summary
            gc.collect()

        representation_summary = pd.concat(batch_summaries, ignore_index=True)
        representation_summary["representation"] = representation
        representation_summary = representation_summary.sort_values("sobol_id")
        representation_summary.to_csv(
            representation_dir / "all_settings_summary.csv", index=False
        )
        all_representation_summaries.append(representation_summary)

        del search_cache
        if representation != "no_pca":
            del C_current
        gc.collect()

    combined_summary = pd.concat(all_representation_summaries, ignore_index=True)
    combined_summary.to_csv(output_root / "all_representations_summary.csv", index=False)
    figures = plot_sobol_search_summaries(
        combined_summary, output_root, species=np.asarray(prof.species),
        show_plots=show_summary_plots,
    )
    print(f"Sobol PCA search complete -> {output_root}")
    return {
        "summary": combined_summary,
        "designs": designs,
        "figures": figures,
        "output_dir": str(output_root),
    }


def fit_nested_pca(
    prof,
    *,
    max_components=500,
    output_dir,
    random_state=0,
    overwrite=False,
):
    """Fit one PCA to standardized ortholog profiles and save reusable artifacts.

    Smaller PCA representations are prefixes of the saved score/component
    matrices, so this fit supports PCA-50, PCA-100, and other nested dimensions.
    """
    from pathlib import Path
    import json
    import numpy as np
    from sklearn.decomposition import PCA

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"PCA output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_genes = len(prof.genes)
    n_features = int(prof.meta["n_ortho"])
    max_components = int(max_components)
    if not 2 <= max_components <= min(n_genes, n_features):
        raise ValueError(
            f"max_components must be between 2 and {min(n_genes, n_features)}"
        )
    if prof.features.shape != (n_genes, n_features):
        raise ValueError("prof feature dimensions disagree with its metadata")

    X = prof.features.astype(np.float32, copy=False)
    X -= X.mean(axis=1, keepdims=True)
    row_sd = X.std(axis=1, keepdims=True)
    X /= np.where(row_sd > 0, row_sd, 1).astype(np.float32)
    model = PCA(
        n_components=max_components,
        svd_solver="randomized",
        whiten=False,
        random_state=random_state,
    )
    scores = model.fit_transform(X).astype(np.float32, copy=False)
    components = model.components_.astype(np.float32, copy=False)

    score_path = output_dir / f"pca_scores_{max_components}.npy"
    component_path = output_dir / f"pca_components_{max_components}.npy"
    np.save(score_path, scores)
    np.save(component_path, components)
    np.save(output_dir / "pca_feature_mean.npy", model.mean_.astype(np.float32))
    np.save(
        output_dir / "pca_explained_variance_ratio.npy",
        model.explained_variance_ratio_,
    )
    configuration = {
        "n_genes": n_genes,
        "n_orthologs": n_features,
        "max_components": max_components,
        "random_state": random_state,
        "whiten": False,
        "row_standardized_before_pca": True,
        "cumulative_explained_variance": float(
            model.explained_variance_ratio_.sum()
        ),
    }
    (output_dir / "pca_fit_configuration.json").write_text(
        json.dumps(configuration, indent=2)
    )
    print(
        f"Saved PCA-{max_components} ({configuration['cumulative_explained_variance']:.2%} "
        f"variance) -> {output_dir}"
    )
    return {
        "scores": scores,
        "components": components,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "configuration": configuration,
        "score_path": str(score_path),
        "component_path": str(component_path),
        "output_dir": str(output_dir),
    }
