# Cross-species clustering pipeline

This directory separates the reproducible mouse–human clustering workflow from
the experiments used to choose and interpret it.

The design for the next hyperparameter-optimization stage is documented in
`docs/objective_landscape_plan.md`.

## Run order

1. Run `notebooks/01_full_pipeline.ipynb` to go from the smoothed datasets and
   homology table to aligned ortholog profiles, the second-order PCC matrix, the
   joint within/cross-species graph, and Leiden modules.
2. Run `notebooks/02_evaluate_selected_clustering.ipynb` to calculate gene
   margin, cross-species integration, cohesion, diversity, and the standard
   diagnostic plots for a saved core run.
3. Use the notebooks under `notebooks/experiments/` only when running a search
   or a representation experiment.

The core notebook is the only notebook required to create a clustering. Its
parameters are collected near the top of the notebook.

## Layout

```text
Cross Species Pipeline/
├── README.md
├── archive/                         exact pre-reorganization notebooks
├── docs/
│   └── objective_landscape_plan.md
├── notebooks/
│   ├── 01_full_pipeline.ipynb       data → joint graph → Leiden modules
│   ├── 02_evaluate_selected_clustering.ipynb
│   └── experiments/
│       ├── 01_grid_search.ipynb
│       ├── 02_pca_experiments.ipynb
│       ├── 03_sobol_search.ipynb
│       └── 04_objective_landscape_search.ipynb
├── src/cross_species/               reusable implementations
│   ├── metrics.py                   expanded scalar objective metric registry
│   ├── objective.py                 desirability calibration and scoring
│   └── optimization.py              search layer (next phase)
└── results/                         outputs made by the new notebooks
```

## Core artifact contract

Each `results/core/core_*` directory is self-contained. It includes the aligned
profiles, gene order, species labels, second-order similarity matrix, sparse
graph, Leiden labels, node/edge tables, ortholog pairs, and a `manifest.json`.
The loader validates dimensions and a hash of the gene/species order before a
downstream notebook can use the run. This prevents a clustering or expression
table from being silently matched to the wrong genes.

The implementation lives in `src/cross_species/`; notebooks import it instead
of redefining long functions in cells. Add the `src` directory to `PYTHONPATH`
when using the package outside these notebooks.

## Results

New outputs are grouped by purpose under `results/`. Historical outputs from
the former `hyperparameter_results/` directory are sorted into each category's
`legacy/` directory. See `results/README.md` for the exact mapping and guidance
on reusing those runs.
