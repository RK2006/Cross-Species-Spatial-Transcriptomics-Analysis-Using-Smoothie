"""Cross-species spatial-transcriptomics clustering toolkit."""

from .alignment import AlignedPair, align_matrices, load_homology, save_aligned
from .artifacts import build_manifest, gene_order_hash, load_core_run, save_core_run
from .evaluation import (
    compare_clustering_evaluations,
    evaluate_clusterings,
    evaluate_joint_modules,
)
from .graph import DualGraph, dual_knn_graph
from .metrics import (
    METRIC_DEFINITIONS,
    METRIC_REGISTRY,
    MetricDefinition,
    metric_metadata,
    metric_vector,
)
from .objective import (
    DesirabilityRule,
    FIRST_PASS_METRICS,
    ObjectiveScore,
    ScoreSpecification,
    educated_full_score_spec,
    fit_first_pass_score_spec,
    score_objective,
    score_objective_table,
)
from .optimization import (
    DEFAULT_PARAMETER_RANGES,
    KEY_HEATMAP_METRICS,
    SOBOL_CHECKPOINTS,
    collect_search_results,
    generate_sobol_settings,
    load_neighbor_cache,
    order_settings_fast_first,
    plot_metric_heatmap,
    prepare_neighbor_cache,
    run_wide_sobol_search,
    save_neighbor_cache,
)
from .representation import EdgeSets, StitchedProfiles, plot_edges, profile_correlation, stitch_profiles
from .search import (
    PARAMS, adjacent_agreement, cache_neighbors, cached_graph,
    review_hyperparameter_experiment, screen, stability,
)

__all__ = [
    "AlignedPair",
    "DualGraph",
    "EdgeSets",
    "DesirabilityRule",
    "DEFAULT_PARAMETER_RANGES",
    "FIRST_PASS_METRICS",
    "KEY_HEATMAP_METRICS",
    "METRIC_DEFINITIONS",
    "METRIC_REGISTRY",
    "MetricDefinition",
    "ObjectiveScore",
    "PARAMS",
    "StitchedProfiles",
    "ScoreSpecification",
    "SOBOL_CHECKPOINTS",
    "adjacent_agreement",
    "align_matrices",
    "build_manifest",
    "cache_neighbors",
    "cached_graph",
    "compare_clustering_evaluations",
    "collect_search_results",
    "dual_knn_graph",
    "evaluate_clusterings",
    "evaluate_joint_modules",
    "educated_full_score_spec",
    "fit_first_pass_score_spec",
    "gene_order_hash",
    "generate_sobol_settings",
    "load_core_run",
    "load_homology",
    "load_neighbor_cache",
    "metric_metadata",
    "metric_vector",
    "order_settings_fast_first",
    "plot_edges",
    "plot_metric_heatmap",
    "profile_correlation",
    "review_hyperparameter_experiment",
    "prepare_neighbor_cache",
    "save_aligned",
    "save_core_run",
    "score_objective",
    "score_objective_table",
    "run_wide_sobol_search",
    "screen",
    "stability",
    "stitch_profiles",
    "save_neighbor_cache",
]
