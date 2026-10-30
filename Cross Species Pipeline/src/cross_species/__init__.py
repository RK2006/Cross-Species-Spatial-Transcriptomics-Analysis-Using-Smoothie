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
from .representation import EdgeSets, StitchedProfiles, plot_edges, profile_correlation, stitch_profiles
from .search import (
    PARAMS, adjacent_agreement, cache_neighbors, cached_graph,
    review_hyperparameter_experiment, screen, stability,
)

__all__ = [
    "AlignedPair",
    "DualGraph",
    "EdgeSets",
    "METRIC_DEFINITIONS",
    "METRIC_REGISTRY",
    "MetricDefinition",
    "PARAMS",
    "StitchedProfiles",
    "adjacent_agreement",
    "align_matrices",
    "build_manifest",
    "cache_neighbors",
    "cached_graph",
    "compare_clustering_evaluations",
    "dual_knn_graph",
    "evaluate_clusterings",
    "evaluate_joint_modules",
    "gene_order_hash",
    "load_core_run",
    "load_homology",
    "metric_metadata",
    "metric_vector",
    "plot_edges",
    "profile_correlation",
    "review_hyperparameter_experiment",
    "save_aligned",
    "save_core_run",
    "screen",
    "stability",
    "stitch_profiles",
]
