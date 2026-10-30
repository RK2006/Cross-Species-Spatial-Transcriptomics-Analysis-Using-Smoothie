# Clustering Objective and Hyperparameter-Landscape Plan

## Goal

Build a general workflow that accepts:

1. A fixed second-order gene-similarity matrix, such as the original
   ortholog-profile PCC matrix or a PCA-derived PCC matrix.
2. A clustering method, initially Leiden.
3. The hyperparameters required to construct and cluster the joint-species
   network.

The workflow will produce gene modules, evaluate those modules with a vector of
scalar metrics, combine the metrics into one documented objective score, and
explore how that score changes throughout hyperparameter space.

The initial global exploration will use a Sobol quasi-random design. Promising
regions can then be sampled more densely to characterize local behavior and
locate strong candidate settings.

## Formal definition

Let:

- \(C\) be a fixed second-order similarity matrix.
- \(x\) be a hyperparameter setting.
- \(G(C,x)\) be the graph built from \(C\) using \(x\).
- \(L(C,x)\) be the module labels produced by Leiden.
- \(y(C,x)\) be the vector of clustering metrics.
- \(S(y)\) be the scalar scoring rule.

The complete objective is:

\[
f_C(x)=S\!\left(y\!\left(C,G(C,x),L(C,x)\right)\right)
\]

For the current graph and clustering method, the initial parameter vector is:

\[
x = \left(
k_{\mathrm{intra}},
k_{\mathrm{inter}},
c_{\mathrm{intra}},
c_{\mathrm{inter}},
r_{\mathrm{Leiden}}
\right)
\]

where \(c\) denotes a PCC cutoff and \(r_{\mathrm{Leiden}}\) is the Leiden
resolution. The random seed is controlled separately and is not an optimization
parameter.

Different representations of \(C\), such as no PCA, PCA-50, and PCA-100, should
be treated as separate optimization landscapes. Their similarity distributions
and effective cutoff ranges can differ, so they should not be interpreted as
one continuous surface.

## Computational stages

### 1. Construct the graph and modules

For one setting \(x\):

```text
C + graph parameters -> joint within/cross-species graph
graph + Leiden parameters -> module labels
```

The output should retain the graph, labels, graph diagnostics, run time,
representation identifier, seed, and complete parameter setting.

### 2. Calculate reusable evaluation tables

The expensive gene-level and module-level quantities should be calculated once
per clustering and stored as reusable tables:

- One row per gene: gene margin, cross-strength fraction, cross-edge presence,
  and reciprocal-neighbor support.
- One row per module: size, species composition, cohesion components, diversity,
  and summarized gene-margin values.
- One row per species and overall: integration and reciprocal-support summaries.

Individual scalar metric functions should extract one value from these tables.
This gives every \(y_i\) its own function without recalculating the same matrices
for every metric.

Conceptually:

```text
evaluate_clustering(...) -> evaluation tables
median_gene_margin(evaluation) -> scalar
p10_gene_margin(evaluation) -> scalar
median_cross_cohesion(evaluation) -> scalar
...
metric_vector(evaluation, metric_spec) -> named vector y
```

Every metric definition should record:

- Name
- Category
- Required input table
- Aggregation rule
- Whether higher, lower, or a target range is preferred
- Valid range and missing-value behavior
- Whether modules or genes receive equal weight

The gene-weighted versus module-weighted distinction must be explicit. A simple
median across modules gives a small module the same influence as a large module,
whereas a gene-weighted summary emphasizes the experience of a typical gene.

### 3. Construct the metric vector

An initial metric vector could contain the following values.

| Category | Candidate scalar metrics |
|---|---|
| Separation | Median gene margin; 10th-percentile gene margin; fraction of negative-margin genes |
| Integration | Median cross-strength fraction; cross-edge coverage; reciprocal gene coverage |
| Module quality | Median mouse cohesion; median human cohesion; median cross-species cohesion; conserved diversity |
| Module structure | Isolated-gene fraction; largest-module fraction; number of meaningful modules |

ARI is excluded from the initial metric vector. It can be added later as a
separate stability analysis if needed.

The stored vector should be a named record, such as one row in a table, rather
than an unlabeled NumPy array. The result table should contain both the raw
metrics and their transformed desirability values.

## From metrics to one score

### Desirability transformations

The raw metrics have different units, ranges, and preferred directions. Each
raw metric should therefore be transformed into a desirability value:

\[
d_i(y_i)\in[0,1]
\]

For a metric where higher values are preferred, one possible transformation is:

\[
d_i(y_i)=
\operatorname{clip}\left(
\frac{y_i-L_i}{T_i-L_i},0,1
\right)
\]

Here, \(L_i\) is an unacceptable value and \(T_i\) is a satisfactory target.
Metrics such as module count or largest-module fraction should use a two-sided
function with an acceptable range.

The initial Sobol sample can help choose plausible thresholds. Once chosen,
these thresholds must be frozen before local refinement so that the definition
of \(f_C(x)\) does not change as new points are added.

### Category scores

Several metrics measure related properties. To avoid giving one concept more
weight merely because it has more metrics, first combine metrics within four
categories:

\[
\begin{aligned}
S_{\mathrm{separation}} &= h(\text{margin metrics}) \\
S_{\mathrm{integration}} &= h(\text{cross-strength and reciprocity metrics}) \\
S_{\mathrm{quality}} &= h(\text{cohesion and diversity metrics}) \\
S_{\mathrm{structure}} &= h(\text{module-size and isolate metrics})
\end{aligned}
\]

### Final objective

A weighted geometric mean is a useful initial scoring rule:

\[
S(y)=
\exp\left[
\frac{\sum_j w_j\log(S_j+\epsilon)}{\sum_j w_j}
\right]
\]

It penalizes settings that perform very poorly in one category. A weighted
arithmetic mean is easier to interpret but permits a very strong category to
fully compensate for a failed category.

The score should also enforce basic feasibility constraints. Examples include:

- Maximum isolated-gene fraction
- Maximum singleton fraction
- Acceptable range for meaningful module count
- Maximum largest-module fraction
- Minimum cross-edge coverage
- Minimum representation of both species in meaningful modules

Both the single score and the full metric vector must be retained. The Pareto
frontier should be inspected alongside the scalar ranking because the final
category weights encode scientific preferences rather than an objective truth.

## Exploring the hyperparameter landscape

The landscape will probably be jagged rather than smooth. A small parameter
change can add or remove graph edges, and Leiden can then reorganize an entire
partition. Gradient-based optimization is therefore inappropriate.

### Global exploration

1. Define representation-specific parameter ranges.
2. Generate a scrambled Sobol design.
3. Round the two \(k\) parameters to valid integers.
4. Evaluate each unique setting and save it immediately.
5. Retain failed and infeasible settings with explicit status codes.
6. Inspect raw metrics, category scores, total scores, and the Pareto frontier.

A small 32-setting design is useful as a pipeline test. Characterizing a
five-dimensional landscape will generally require a larger design, such as
128-512 settings, depending on run time and the observed complexity.

### Landscape approximation

Fit a surrogate model from \(x\) to each important output:

- Total score
- Each category score
- Selected raw metrics

A random-forest or boosted-tree surrogate is a reasonable first choice because
it can model nonlinear thresholds and interactions without assuming a smooth
response. Its predictive accuracy should be assessed by cross-validation before
interpreting the surface.

Useful visualizations include:

- Score distribution and rank plot
- Parallel-coordinate plot of parameters and category scores
- One-parameter partial-dependence or accumulated-local-effect plots
- Two-parameter response heatmaps with other parameters fixed or averaged
- Observed-versus-predicted surrogate diagnostics
- Pareto-front plots for competing categories
- Local slices through a promising setting
- Graph-constraint diagnostics showing whether \(k\) or the PCC cutoff was
  binding

### Local refinement

1. Select several promising and scientifically distinct regions rather than
   only the single highest observed score.
2. Narrow the parameter ranges around each region.
3. Generate another Sobol design within those ranges.
4. Re-evaluate and update the surrogate.
5. Compare finalists using the complete metric vector and module-level plots.

## Guarding against a misleading objective

Several metrics are calculated from the same matrix used to construct the
graph. Optimizing them can reward trivial behavior:

- Many small modules can inflate margin and cohesion.
- A single large module can inflate apparent species integration.
- Closely related margin statistics can unintentionally dominate the score.
- A setting can appear optimal because its cutoff is inactive and \(k\) alone
  controls the graph.

Structure penalties, graph-constraint diagnostics, and metrics based on spatial
expression should therefore accompany the internal similarity-based
metrics. Final biological review should remain separate from the mathematical
objective used during the search.

## Proposed code and notebook organization

This work should use a separate notebook:

```text
notebooks/experiments/04_objective_landscape_search.ipynb
```

The notebook should orchestrate an experiment and display results. Reusable
calculations should remain in source modules:

```text
src/cross_species/
├── evaluation.py       existing gene/module evaluation tables
├── metrics.py          one documented scalar function per y_i and a registry
├── objective.py        desirability transforms and category/final scores
└── optimization.py     Sobol designs, resumable runs, surrogates, refinement
```

The notebook should contain these sections:

1. Load and validate a core run.
2. Select one representation matrix \(C\).
3. Define the Leiden and graph parameter space.
4. Select the metric registry and aggregation rules.
5. Define or load a frozen scoring specification.
6. Generate and run the Sobol design.
7. Inspect raw metrics and feasibility failures.
8. Calculate category and total scores.
9. Inspect Pareto fronts and landscape plots.
10. Define and run local refinement regions.
11. Export the shortlist and complete provenance.

Results should be written to a timestamped directory:

```text
results/optimization/objective_search_YYYYMMDD_HHMMSS/
├── configuration.json
├── metric_specification.json
├── scoring_specification.json
├── settings.csv
├── raw_metrics.csv
├── scores.csv
├── failures.csv
├── labels/
├── surrogate/
└── plots/
```

Each run must record the gene-order hash, representation identifier, input
matrix location, algorithm and package versions, parameter ranges, Sobol seed,
Leiden seed, metric definitions, scoring thresholds, and category weights.

## Implementation sequence

1. Finalize the initial scalar metrics and clarify gene-weighted versus
   module-weighted aggregation.
2. Refactor the existing evaluator so expensive tables are computed once and
   scalar metric functions consume those tables.
3. Implement the named metric registry and test every metric on one saved
   clustering.
4. Define feasibility rules and desirability transformations.
5. Implement category scores and the final score.
6. Create the separate objective-landscape notebook.
7. Run a small Sobol smoke test to verify artifacts and resume behavior.
8. Run the larger global design.
9. Freeze the scoring specification and inspect the landscape.
10. Perform local refinement and inspect the finalists.

## Decisions needed before implementation

- Exact scalar metrics included in the first version of \(y\)
- Gene-weighted or module-weighted summaries for each module metric
- Hard feasibility constraints versus soft penalties
- Desirability thresholds and category weights
- Initial Sobol size and computational budget
- Number and definition of local refinement regions
