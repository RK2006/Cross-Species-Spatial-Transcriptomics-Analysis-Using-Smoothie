# Result directories

The reorganized notebooks write new artifacts here:

- `core/`: reusable clustering runs with a manifest and validated gene order
- `evaluation/`: full metric tables and plots for selected clusterings
- `grid_search/`: regular grid screens, stability runs, and batched evaluations
- `pca/`: nested PCA artifacts, similarity calibration, and PC interpretation
- `sobol/`: 32-setting representation-specific Sobol searches
- `optimization/`: wide objective-vector searches, raw metric tables, and heatmaps
- `cache/`: reusable ranked-neighbor arrays keyed to a validated gene order

The wide no-PCA experiment writes to
`optimization/wide_sobol_no_pca_512/`. Each setting has an atomic JSON
checkpoint and a compressed label vector. The consolidated `raw_metrics.csv`
retains all 65 unmodified indicators; heatmap normalization is display-only.

Timestamped run directories prevent one experiment from overwriting another.
Large generated contents in these folders are ignored by Git; each folder's
`.gitkeep` file keeps the layout visible in a fresh checkout.

Historical output from the former `hyperparameter_results/` directory is sorted
without changing file contents:

- `grid_search/legacy/20260908_111238_447833/`: the original 81-setting screen
  and seed/resolution stability results
- `evaluation/legacy/module_evaluation_20261009_073527/`: the 27 evaluation
  batches and hyperparameter reviews
- `pca/legacy/pca_pilot_20261014_223852/`: the PCA-50/100/200/500 pilot
- `pca/legacy/pca_similarity_calibration_50_100_*/`: both saved similarity
  calibration variants

These runs do not necessarily contain the manifest used by `load_core_run`.
Their evaluation tables remain compatible with the legacy review helper. Create
a new core run before using the reorganized downstream notebooks. Paths recorded
inside legacy CSV and JSON manifests still show their original
`hyperparameter_results/...` locations; the files themselves remain unchanged.
