"""Evaluation and comparison of joint-species gene clusterings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse



def evaluate_joint_modules(
    labels,
    C,
    G,
    prof,
    *,
    expression_by_species=None,
    k_inter=None,
    cutoff_inter=None,
    min_module_size=2,
    diversity_min_genes=2,
    matrix_block=512,
    expression_block=32,
    return_module_scores=False,
    make_plots=False,
    show_plots=True,
    plot_min_module_size=10,
):
    """Evaluate one joint-species gene clustering.

    Parameters
    ----------
    labels
        One module label per gene, in the same order as ``C`` and ``prof``.
    C
        Dense gene-by-gene second-order similarity matrix. It is used for
        module cohesion, split into within- and cross-species components.
    G
        ``DualGraph`` actually clustered. Its raw sparse adjacency weights are
        used for gene margin and cross-strength fractions. Missing edges count
        as zero in the Smoothie-style margin calculation.
    prof
        ``StitchedProfiles`` supplying tagged names, raw names, and species.
    expression_by_species
        Optional mapping from each species label to an AnnData object, a
        spots-by-genes DataFrame, or an ``(X, gene_names)`` tuple. When given,
        spatial module scores and module diversity are computed separately in
        each species. Each gene is divided by its maximum before averaging.
    k_inter, cutoff_inter
        Cross-neighbor rule used to build the graph. These are needed to
        reconstruct directed proposals and identify reciprocal cross-neighbors.
        They default to values in ``G.meta`` when available.
    min_module_size
        Modules smaller than this have undefined gene margins and are excluded
        as alternative modules in the margin calculation.
    diversity_min_genes
        Minimum matched, nonzero genes required to construct a species-specific
        module score.
    make_plots
        Generate the seven diagnostic figures requested for this analysis.
    show_plots
        Display generated figures immediately. Figures are always returned.
    plot_min_module_size
        Minimum total module size for module-level scatterplots.

    Returns
    -------
    dict
        ``genes``: one row per gene with margin and integration metrics.
        ``modules``: size, three cohesion components, and diversity metrics.
        ``integration``: reciprocal-support summaries by species and overall.
        ``summary``: compact clustering-level metric summary.
        ``module_scores``: optional spatial score matrices when requested.
        ``figures``: mapping of plot names to Matplotlib Figure objects.

    Notes
    -----
    * Margin evaluates the exact sparse, raw-weight graph passed to Leiden.
    * Cohesion evaluates dense C and is therefore an internal validation score.
    * Diversity is species-specific because mouse and human spatial coordinates
      are not aligned. Conserved diversity is the minimum across species.
    """
    labels = np.asarray(labels)
    C = np.asarray(C)
    species = np.asarray(prof.species)
    genes = np.asarray(prof.genes)
    raw_genes = np.asarray(prof.raw_genes)
    W = G.adj.tocsr().astype(float, copy=False)
    n = len(labels)

    if C.shape != (n, n) or W.shape != (n, n):
        raise ValueError("labels, C, and G must describe the same number of genes")
    if any(len(x) != n for x in (species, genes, raw_genes)):
        raise ValueError("prof gene metadata does not match labels")
    species_values = pd.unique(species)
    if len(species_values) != 2:
        raise ValueError("this evaluator currently requires exactly two species")
    species_lookup = {str(value).lower(): value for value in species_values}
    if {"mouse", "human"}.issubset(species_lookup):
        species_values = np.asarray([species_lookup["mouse"], species_lookup["human"]])
    if min_module_size < 2 or diversity_min_genes < 2 or plot_min_module_size < 2:
        raise ValueError("minimum module sizes must be at least 2")
    if matrix_block < 1 or expression_block < 1:
        raise ValueError("block sizes must be positive")
    if W.nnz and np.nanmin(W.data) < 0:
        raise ValueError("gene margin requires nonnegative graph weights")

    module_values, module_index = np.unique(labels, return_inverse=True)
    n_modules = len(module_values)
    module_sizes = np.bincount(module_index, minlength=n_modules)
    membership = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), module_index)), shape=(n, n_modules)
    )
    eligible = module_sizes >= min_module_size

    # Smoothie-style similarity silhouette on the graph actually clustered.
    own_affinity = np.full(n, np.nan)
    alternative_affinity = np.full(n, np.nan)
    closest_index = np.full(n, -1, dtype=int)
    for start in range(0, n, matrix_block):
        stop = min(start + matrix_block, n)
        rows = np.arange(start, stop)
        sums = (W[start:stop] @ membership).toarray()
        means = sums / module_sizes[None, :]
        own = module_index[rows]
        own_denominator = module_sizes[own] - 1
        valid_own = own_denominator > 0
        own_affinity[rows[valid_own]] = (
            sums[np.arange(len(rows))[valid_own], own[valid_own]]
            / own_denominator[valid_own]
        )
        means[:, ~eligible] = -np.inf
        means[np.arange(len(rows)), own] = -np.inf
        best = np.argmax(means, axis=1)
        best_value = means[np.arange(len(rows)), best]
        valid_best = np.isfinite(best_value)
        alternative_affinity[rows[valid_best]] = best_value[valid_best]
        closest_index[rows[valid_best]] = best[valid_best]

    margin_denominator = np.maximum(own_affinity, alternative_affinity)
    margin = np.full(n, np.nan)
    valid_margin = np.isfinite(margin_denominator) & (margin_denominator > 0)
    margin[valid_margin] = (
        own_affinity[valid_margin] - alternative_affinity[valid_margin]
    ) / margin_denominator[valid_margin]
    closest_module = np.full(n, None, dtype=object)
    has_closest = closest_index >= 0
    closest_module[has_closest] = module_values[closest_index[has_closest]]

    # Per-gene cross-species strength and degree on the final undirected graph.
    coo = W.tocoo()
    is_cross = species[coo.row] != species[coo.col]
    total_strength = np.asarray(W.sum(axis=1)).ravel()
    total_degree = np.asarray(W.getnnz(axis=1)).ravel()
    cross_strength = np.bincount(
        coo.row[is_cross], weights=coo.data[is_cross], minlength=n
    )
    cross_degree = np.bincount(coo.row[is_cross], minlength=n)
    cross_strength_fraction = np.full(n, np.nan)
    active = total_strength > 0
    cross_strength_fraction[active] = cross_strength[active] / total_strength[active]

    # Reconstruct directed cross-neighbor proposals to measure reciprocity.
    if k_inter is None:
        k_inter = G.meta.get("k_inter")
    if cutoff_inter is None:
        cutoff_inter = G.meta.get("cutoff_inter")
    if k_inter is None or cutoff_inter is None:
        raise ValueError(
            "provide k_inter and cutoff_inter, or store them in G.meta"
        )
    if k_inter < 1:
        raise ValueError("k_inter must be positive")

    proposal_rows, proposal_cols = [], []
    for source_species in species_values:
        source = np.flatnonzero(species == source_species)
        target = np.flatnonzero(species != source_species)
        kk = min(int(k_inter), len(target))
        for start in range(0, len(source), matrix_block):
            rr = source[start:start + matrix_block]
            values = np.array(C[np.ix_(rr, target)], dtype=float, copy=True)
            values[~np.isfinite(values)] = -np.inf
            part = np.argpartition(-values, kk - 1, axis=1)[:, :kk]
            weights = np.take_along_axis(values, part, axis=1)
            keep = weights >= cutoff_inter
            local_row, local_col = np.where(keep)
            proposal_rows.append(rr[local_row])
            proposal_cols.append(target[part[local_row, local_col]])

    proposal_rows = np.concatenate(proposal_rows)
    proposal_cols = np.concatenate(proposal_cols)
    proposals = sparse.csr_matrix(
        (np.ones(len(proposal_rows), dtype=np.int8),
         (proposal_rows, proposal_cols)),
        shape=(n, n),
    )
    proposals.sum_duplicates()
    proposals.data[:] = 1
    reciprocal = proposals.multiply(proposals.T).astype(bool).tocsr()
    cross_union = proposals.maximum(proposals.T).astype(bool).tocsr()
    reciprocal_degree = np.asarray(reciprocal.getnnz(axis=1)).ravel()
    proposed_cross_degree = np.asarray(cross_union.getnnz(axis=1)).ravel()
    reciprocal_fraction = np.full(n, np.nan)
    has_cross_proposal = proposed_cross_degree > 0
    reciprocal_fraction[has_cross_proposal] = (
        reciprocal_degree[has_cross_proposal]
        / proposed_cross_degree[has_cross_proposal]
    )

    gene_metrics = pd.DataFrame({
        "gene": genes,
        "raw_gene": raw_genes,
        "species": species,
        "module": labels,
        "module_size": module_sizes[module_index],
        "own_module_affinity": own_affinity,
        "best_other_affinity": alternative_affinity,
        "closest_other_module": closest_module,
        "gene_margin": margin,
        "degree": total_degree,
        "strength": total_strength,
        "cross_degree": cross_degree,
        "cross_strength": cross_strength,
        "cross_strength_fraction": cross_strength_fraction,
        "reciprocal_cross_degree": reciprocal_degree,
        "reciprocal_cross_fraction": reciprocal_fraction,
    })

    # Dense second-order cohesion: overall, species A-A, species B-B, and A-B.
    def mean_off_diagonal(matrix):
        size = matrix.shape[0]
        if size < 2:
            return np.nan
        diagonal = np.diag(matrix)
        finite_count = np.isfinite(matrix).sum() - np.isfinite(diagonal).sum()
        if finite_count == 0:
            return np.nan
        return float((np.nansum(matrix) - np.nansum(diagonal)) / finite_count)

    def mean_rectangle(matrix):
        return float(np.nanmean(matrix)) if matrix.size else np.nan

    module_rows = []
    first_species, second_species = species_values
    for module_number, module_label in enumerate(module_values):
        idx = np.flatnonzero(module_index == module_number)
        idx_first = idx[species[idx] == first_species]
        idx_second = idx[species[idx] == second_species]
        module_rows.append({
            "module": module_label,
            "n_genes": len(idx),
            f"n_{first_species}": len(idx_first),
            f"n_{second_species}": len(idx_second),
            "cohesion_all": mean_off_diagonal(C[np.ix_(idx, idx)]),
            f"cohesion_{first_species}_{first_species}":
                mean_off_diagonal(C[np.ix_(idx_first, idx_first)]),
            f"cohesion_{second_species}_{second_species}":
                mean_off_diagonal(C[np.ix_(idx_second, idx_second)]),
            f"cohesion_{first_species}_{second_species}":
                mean_rectangle(C[np.ix_(idx_first, idx_second)]),
            "median_gene_margin": float(np.nanmedian(margin[idx]))
                if np.isfinite(margin[idx]).any() else np.nan,
            "fraction_negative_margin": float(
                np.mean(margin[idx][np.isfinite(margin[idx])] < 0)
            ) if np.isfinite(margin[idx]).any() else np.nan,
        })
    module_metrics = pd.DataFrame(module_rows)

    # Species-specific spatial module scores and nearest-module diversity.
    score_outputs = {}
    diversity_columns = []
    if expression_by_species is not None:
        missing_species = set(species_values) - set(expression_by_species)
        if missing_species:
            raise ValueError(f"missing expression data for {sorted(missing_species)}")

        def unpack_expression(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_numpy(), np.asarray(obj.columns, dtype=str)
            if isinstance(obj, tuple) and len(obj) == 2:
                return obj[0], np.asarray(obj[1], dtype=str)
            if hasattr(obj, "X") and hasattr(obj, "var_names"):
                return obj.X, np.asarray(obj.var_names, dtype=str)
            raise TypeError(
                "expression values must be AnnData, a spots-by-genes DataFrame, "
                "or an (X, gene_names) tuple"
            )

        for species_name in species_values:
            X, expression_genes = unpack_expression(
                expression_by_species[species_name]
            )
            exact = {name: i for i, name in enumerate(expression_genes)}
            upper_groups = {}
            for i, name in enumerate(expression_genes):
                upper_groups.setdefault(name.upper(), []).append(i)

            score_list, score_modules, score_gene_counts = [], [], []
            for module_number, module_label in enumerate(module_values):
                gene_rows = np.flatnonzero(
                    (module_index == module_number) & (species == species_name)
                )
                columns = []
                for name in raw_genes[gene_rows].astype(str):
                    if name in exact:
                        columns.append(exact[name])
                    elif len(upper_groups.get(name.upper(), [])) == 1:
                        columns.append(upper_groups[name.upper()][0])
                columns = np.asarray(columns, dtype=int)
                if len(columns) < diversity_min_genes:
                    continue

                score_sum = np.zeros(X.shape[0], dtype=np.float64)
                used = 0
                for start in range(0, len(columns), expression_block):
                    cols = columns[start:start + expression_block]
                    values = X[:, cols]
                    if sparse.issparse(values):
                        maxima_raw = values.max(axis=0)
                        maxima = np.asarray(
                            maxima_raw.toarray()
                            if sparse.issparse(maxima_raw) else maxima_raw
                        ).ravel()
                    else:
                        values = np.asarray(values)
                        maxima = np.nanmax(values, axis=0)
                    valid = np.isfinite(maxima) & (maxima > 0)
                    if not valid.any():
                        continue
                    inverse_max = 1.0 / maxima[valid]
                    if sparse.issparse(values):
                        contribution = values[:, valid].dot(inverse_max)
                    else:
                        contribution = values[:, valid] @ inverse_max
                    score_sum += np.asarray(contribution).ravel()
                    used += int(valid.sum())
                if used < diversity_min_genes:
                    continue
                score_list.append((score_sum / used).astype(np.float32))
                score_modules.append(module_label)
                score_gene_counts.append(used)

            diversity = np.full(len(score_modules), np.nan)
            nearest = np.full(len(score_modules), None, dtype=object)
            if len(score_list) >= 2:
                score_matrix = np.column_stack(score_list)
                correlations = np.corrcoef(score_matrix, rowvar=False)
                np.fill_diagonal(correlations, np.nan)
                for i in range(len(score_modules)):
                    finite = np.isfinite(correlations[i])
                    if finite.any():
                        candidates = np.flatnonzero(finite)
                        best = candidates[np.argmax(correlations[i, candidates])]
                        diversity[i] = 1.0 - correlations[i, best]
                        nearest[i] = score_modules[best]
            elif score_list:
                score_matrix = np.column_stack(score_list)
            else:
                score_matrix = np.empty((X.shape[0], 0), dtype=np.float32)

            tag = str(species_name)
            diversity_table = pd.DataFrame({
                "module": score_modules,
                f"module_score_genes_{tag}": score_gene_counts,
                f"nearest_module_{tag}": nearest,
                f"diversity_{tag}": diversity,
            })
            module_metrics = module_metrics.merge(
                diversity_table, on="module", how="left"
            )
            diversity_columns.append(f"diversity_{tag}")
            if return_module_scores:
                score_outputs[tag] = {
                    "scores": score_matrix,
                    "modules": np.asarray(score_modules),
                }

        module_metrics["conserved_diversity"] = module_metrics[
            diversity_columns
        ].min(axis=1, skipna=False)

    # Reciprocal support summaries. Fractions count undirected cross edges once.
    union_edges = sparse.triu(cross_union, k=1).nnz
    reciprocal_edges = sparse.triu(reciprocal, k=1).nnz
    integration_rows = []
    for species_name in [*species_values, "overall"]:
        mask = np.ones(n, dtype=bool) if species_name == "overall" else species == species_name
        connected = proposed_cross_degree[mask] > 0
        integration_rows.append({
            "species": species_name,
            "n_genes": int(mask.sum()),
            "cross_edge_coverage": float(connected.mean()),
            "reciprocal_gene_coverage": float((reciprocal_degree[mask] > 0).mean()),
            "median_cross_strength_fraction": float(
                np.nanmedian(cross_strength_fraction[mask])
            ),
            "median_reciprocal_fraction_among_cross_connected": float(
                np.nanmedian(reciprocal_fraction[mask])
            ),
            "reciprocal_cross_edge_fraction":
                reciprocal_edges / union_edges if union_edges else np.nan,
        })
    integration = pd.DataFrame(integration_rows)

    finite_margin = margin[np.isfinite(margin)]
    summary = pd.Series({
        "n_genes": n,
        "n_modules": n_modules,
        "n_margin_genes": len(finite_margin),
        "median_gene_margin": np.median(finite_margin)
            if len(finite_margin) else np.nan,
        "p10_gene_margin": np.quantile(finite_margin, 0.10)
            if len(finite_margin) else np.nan,
        "fraction_negative_margin": np.mean(finite_margin < 0)
            if len(finite_margin) else np.nan,
        "cross_edge_coverage": np.mean(proposed_cross_degree > 0),
        "reciprocal_gene_coverage": np.mean(reciprocal_degree > 0),
        "reciprocal_cross_edge_fraction":
            reciprocal_edges / union_edges if union_edges else np.nan,
        "median_cross_strength_fraction":
            np.nanmedian(cross_strength_fraction),
    }, name="value")

    figures = {}
    if make_plots:
        if expression_by_species is None:
            raise ValueError(
                "make_plots=True requires expression_by_species so the "
                "mouse-versus-human diversity plot can be calculated"
            )

        colors = {
            species_values[0]: "#E1812C",
            species_values[1]: "#3274A1",
        }

        def scaled_marker_sizes(values, low=25, high=220):
            values = np.asarray(values, dtype=float)
            if len(values) == 0:
                return values
            root = np.sqrt(np.maximum(values, 0))
            span = root.max() - root.min()
            if span == 0:
                return np.full(len(values), (low + high) / 2)
            return low + (high - low) * (root - root.min()) / span

        def add_identity_line(axis, x, y):
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                return
            lower = min(np.min(x[finite]), np.min(y[finite]))
            upper = max(np.max(x[finite]), np.max(y[finite]))
            if lower == upper:
                lower -= 0.05
                upper += 0.05
            axis.plot([lower, upper], [lower, upper], ls="--", lw=1, color="0.45")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)

        # 1. Gene-margin histogram across all genes.
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
        margin_values = gene_metrics["gene_margin"].dropna().to_numpy()
        if len(margin_values):
            margin_lower = float(np.quantile(margin_values, 0.005))
            margin_upper = float(np.max(margin_values))
            margin_span = max(margin_upper - margin_lower, 0.05)
            margin_lower = max(-1.0, margin_lower - 0.02 * margin_span)
            margin_upper = min(1.0, margin_upper + 0.02 * margin_span)
            displayed = margin_values[
                (margin_values >= margin_lower) & (margin_values <= margin_upper)
            ]
            margin_bins = np.linspace(margin_lower, margin_upper, 41)
            ax.hist(
                displayed, bins=margin_bins,
                weights=np.full(len(displayed), 1.0 / len(displayed)),
                color="#4C72B0", edgecolor="white", linewidth=0.5,
            )
            ax.set_xlim(margin_lower, margin_upper)
            n_trimmed = int((margin_values < margin_lower).sum())
            if n_trimmed:
                ax.text(
                    0.02, 0.96, f"{n_trimmed:,} lower-tail genes outside view",
                    transform=ax.transAxes, va="top", fontsize=8, color="0.35",
                )
        ax.axvline(0, color="#B22222", ls="--", lw=1)
        ax.set(
            xlabel="Gene margin", ylabel="Fraction of genes per bin",
            title="Gene-margin distribution",
        )
        fig.tight_layout()
        figures["gene_margin_histogram"] = fig

        # 2. Ranked module-size distribution.
        ranked_sizes = np.sort(module_metrics["n_genes"].to_numpy())[::-1]
        ranks = np.arange(1, len(ranked_sizes) + 1)
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
        ax.plot(ranks, ranked_sizes, color="#4C72B0", lw=1.5)
        ax.scatter(ranks, ranked_sizes, color="#4C72B0", s=10, alpha=0.7)
        ax.axhline(
            plot_min_module_size, color="#B22222", ls="--", lw=1,
            label=f"interpretation threshold = {plot_min_module_size}",
        )
        ax.set_yscale("log")
        ax.set(
            xlabel="Module rank", ylabel="Genes in module (log scale)",
            title="Ranked module-size distribution",
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        figures["ranked_module_sizes"] = fig

        # 3. Within-species cohesion comparison; color is cross cohesion.
        x_col = f"cohesion_{first_species}_{first_species}"
        y_col = f"cohesion_{second_species}_{second_species}"
        cross_col = f"cohesion_{first_species}_{second_species}"
        keep = (
            (module_metrics["n_genes"] >= plot_min_module_size)
            & module_metrics[[x_col, y_col, cross_col]].notna().all(axis=1)
        )
        plotted = module_metrics.loc[keep]
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=130)
        points = ax.scatter(
            plotted[x_col], plotted[y_col],
            s=scaled_marker_sizes(plotted["n_genes"]),
            c=plotted[cross_col], cmap="viridis", alpha=0.8,
            edgecolor="white", linewidth=0.4,
        )
        add_identity_line(
            ax, plotted[x_col].to_numpy(), plotted[y_col].to_numpy()
        )
        ax.set(
            xlabel=f"{first_species}–{first_species} cohesion",
            ylabel=f"{second_species}–{second_species} cohesion",
            title="Within-species module cohesion",
        )
        if len(plotted):
            fig.colorbar(points, ax=ax, label=f"{first_species}–{second_species} cohesion")
        fig.tight_layout()
        figures["within_species_cohesion"] = fig

        # 4. Species-specific spatial diversity comparison.
        diversity_first = f"diversity_{first_species}"
        diversity_second = f"diversity_{second_species}"
        diversity_keep = (
            (module_metrics["n_genes"] >= plot_min_module_size)
            & module_metrics[[diversity_first, diversity_second]].notna().all(axis=1)
        )
        plotted = module_metrics.loc[diversity_keep]
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=130)
        points = ax.scatter(
            plotted[diversity_first], plotted[diversity_second],
            s=scaled_marker_sizes(plotted["n_genes"]),
            c=plotted["conserved_diversity"], cmap="plasma", alpha=0.8,
            edgecolor="white", linewidth=0.4,
        )
        add_identity_line(
            ax, plotted[diversity_first].to_numpy(),
            plotted[diversity_second].to_numpy(),
        )
        ax.set(
            xlabel=f"{first_species} diversity",
            ylabel=f"{second_species} diversity",
            title="Spatial module diversity",
        )
        if len(plotted):
            fig.colorbar(points, ax=ax, label="Conserved diversity")
        fig.tight_layout()
        figures["spatial_diversity"] = fig

        # 5. Cross-strength fraction distribution by species.
        violin_values = [
            gene_metrics.loc[
                gene_metrics["species"] == species_name,
                "cross_strength_fraction",
            ].dropna().to_numpy()
            for species_name in species_values
        ]
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=130)
        violin = ax.violinplot(
            violin_values, positions=np.arange(1, len(species_values) + 1),
            showmeans=False, showmedians=True, showextrema=False,
        )
        for body, species_name in zip(violin["bodies"], species_values):
            body.set_facecolor(colors[species_name])
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        violin["cmedians"].set_color("black")
        ax.set_xticks(np.arange(1, len(species_values) + 1), species_values)
        ax.set(
            ylabel="Cross-strength fraction",
            title="Cross-species edge-weight contribution", ylim=(-0.02, 1.02),
        )
        fig.tight_layout()
        figures["cross_strength_by_species"] = fig

        # 6. Reciprocal fraction versus cross strength, faceted by species.
        fig, axes = plt.subplots(
            1, len(species_values), figsize=(10.8, 4.5), dpi=130,
            sharex=True, sharey=True,
        )
        density_maps = ["Oranges", "Blues"]
        for ax, species_name, color_map in zip(
            np.atleast_1d(axes), species_values, density_maps
        ):
            mask = (
                (gene_metrics["species"] == species_name)
                & gene_metrics["cross_strength_fraction"].notna()
                & gene_metrics["reciprocal_cross_fraction"].notna()
            )
            x_values = gene_metrics.loc[mask, "cross_strength_fraction"]
            y_values = gene_metrics.loc[mask, "reciprocal_cross_fraction"]
            if len(x_values):
                density = ax.hexbin(
                    x_values, y_values, gridsize=35, extent=(0, 1, 0, 1),
                    mincnt=1, bins="log", cmap=color_map,
                )
                fig.colorbar(
                    density, ax=ax, label="Gene count (log color scale)"
                )
            ax.set(
                xlabel="Cross-strength fraction",
                title=f"{species_name} (n={len(x_values):,})",
                xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
            )
        np.atleast_1d(axes)[0].set_ylabel(
            "Reciprocal cross-neighbor fraction"
        )
        fig.suptitle("Cross strength versus reciprocal support", y=1.02)
        fig.tight_layout()
        figures["reciprocity_vs_cross_strength"] = fig

        # 7. Module composition versus cross-species cohesion.
        first_fraction = (
            module_metrics[f"n_{first_species}"] / module_metrics["n_genes"]
        )
        composition_keep = (
            (module_metrics["n_genes"] >= plot_min_module_size)
            & module_metrics[cross_col].notna()
        )
        plotted = module_metrics.loc[composition_keep]
        plotted_fraction = first_fraction.loc[composition_keep]
        fig, ax = plt.subplots(figsize=(6.7, 5.2), dpi=130)
        points = ax.scatter(
            plotted_fraction, plotted[cross_col],
            s=scaled_marker_sizes(plotted["n_genes"]),
            c=plotted["median_gene_margin"], cmap="coolwarm",
            alpha=0.82, edgecolor="white", linewidth=0.4,
        )
        ax.axvline(0.5, color="0.45", ls="--", lw=1)
        ax.set(
            xlabel=f"Fraction {first_species} genes in module",
            ylabel=f"{first_species}–{second_species} cohesion",
            title="Module composition versus cross-species cohesion",
            xlim=(-0.02, 1.02),
        )
        if len(plotted):
            fig.colorbar(points, ax=ax, label="Median gene margin")
        fig.tight_layout()
        figures["composition_vs_cross_cohesion"] = fig

        if show_plots:
            plt.show()

    return {
        "genes": gene_metrics,
        "modules": module_metrics,
        "integration": integration,
        "summary": summary,
        "module_scores": score_outputs,
        "figures": figures,
    }



def compare_clustering_evaluations(
    evaluations,
    *,
    min_module_size=10,
    make_plots=False,
    show_plots=True,
    reference=None,
    include_pairwise=True,
    plot_clusterings=None,
):
    """Compare outputs from ``evaluate_joint_modules`` across algorithms.

    Parameters
    ----------
    evaluations
        Mapping ``algorithm_name -> evaluation dictionary`` returned by
        ``evaluate_joint_modules``. Gene tables are aligned by their tagged
        ``gene`` names, so algorithm label numbers do not need to match.
    min_module_size
        Threshold used to define a module as large enough for the meaningful-
        module summaries and restricted pairwise comparisons.
    make_plots
        Generate ranked module sizes, gene-margin and cross-strength violins,
        cohesion and diversity box plots, and agreement heatmaps.
    show_plots
        Display generated figures immediately. Figures are always returned.
    reference
        Name of the reference clustering for module-overlap heatmaps. The
        first clustering is used by default.
    include_pairwise
        Calculate pairwise partition statistics and module-overlap matrices.
        Disable this for large screens to avoid quadratic comparisons.
    plot_clusterings
        Optional clustering names to include in distribution plots. This lets
        a large evaluated screen be displayed in groups of four without
        recalculating the individual evaluations.

    Returns
    -------
    dict
        ``summary`` has one row per algorithm; ``pairwise`` compares whole
        partitions; ``module_matches`` gives the best Jaccard match for
        every sufficiently large module in both comparison directions;
        ``overlap_matrices`` contains full pairwise Jaccard matrices; and
        ``figures`` contains the optional Matplotlib figures.

    Notes
    -----
    Seed stability is intentionally absent: deterministic algorithms require
    data/edge perturbations rather than repeated identical runs. Graph-level
    statistics are included for context but only isolate algorithm effects when
    every evaluation used the same graph and gene universe.
    """
    from itertools import combinations
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        mutual_info_score,
        normalized_mutual_info_score,
    )

    if not isinstance(evaluations, dict) or len(evaluations) < 2:
        raise ValueError("provide at least two named evaluation dictionaries")
    if min_module_size < 2:
        raise ValueError("min_module_size must be at least 2")
    if reference is None:
        reference = next(iter(evaluations))
    if reference not in evaluations:
        raise ValueError(f"unknown reference clustering {reference!r}")

    required_gene_columns = {
        "gene", "species", "module", "module_size", "gene_margin",
        "degree", "strength", "cross_degree", "cross_strength",
        "cross_strength_fraction", "reciprocal_cross_fraction",
    }
    required_module_columns = {"module", "n_genes"}

    def weighted_mean(values, weights):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not keep.any():
            return np.nan
        return float(np.average(values[keep], weights=weights[keep]))

    def finite_quantile(values, q):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.quantile(values, q)) if len(values) else np.nan

    def label_entropy(values):
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log(probabilities)).sum())

    def variation_of_information(a, b):
        return label_entropy(a) + label_entropy(b) - 2 * mutual_info_score(a, b)

    def normalized_species_entropy(frame):
        counts = frame.groupby(["module", "species"]).size().unstack(fill_value=0)
        probabilities = counts.div(counts.sum(axis=1), axis=0)
        number_species = counts.shape[1]
        if number_species < 2:
            return pd.Series(0.0, index=counts.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = probabilities.where(probabilities > 0)
            entropy = -(terms * np.log(terms)).sum(axis=1) / np.log(number_species)
        return entropy.fillna(0.0)

    prepared = {}
    summary_rows = []
    gene_universes = []

    for algorithm, evaluation in evaluations.items():
        if not isinstance(evaluation, dict):
            raise TypeError(f"evaluation for {algorithm!r} must be a dictionary")
        genes = evaluation["genes"].copy()
        modules = evaluation["modules"].copy()
        missing_gene = required_gene_columns - set(genes.columns)
        missing_module = required_module_columns - set(modules.columns)
        if missing_gene or missing_module:
            raise ValueError(
                f"{algorithm!r} missing gene columns {sorted(missing_gene)} "
                f"or module columns {sorted(missing_module)}"
            )
        if genes["gene"].duplicated().any():
            raise ValueError(f"{algorithm!r} contains duplicate tagged gene names")
        genes = genes.set_index("gene", drop=False)
        prepared[algorithm] = {"genes": genes, "modules": modules}
        gene_universes.append(set(genes.index))

        n_genes = len(genes)
        module_sizes = modules["n_genes"].to_numpy()
        meaningful = modules["n_genes"] >= min_module_size
        meaningful_labels = set(modules.loc[meaningful, "module"])
        gene_meaningful = genes["module"].isin(meaningful_labels)
        active = genes["degree"] > 0
        entropy = normalized_species_entropy(genes.reset_index(drop=True))
        mixed = entropy > 0
        mixed_labels = set(entropy.index[mixed])

        row = {
            "algorithm": algorithm,
            "n_genes": n_genes,
            "n_modules": len(modules),
            "n_non_singleton_modules": int((module_sizes >= 2).sum()),
            f"n_modules_ge_{min_module_size}": int(meaningful.sum()),
            f"fraction_genes_in_modules_ge_{min_module_size}":
                float(gene_meaningful.mean()),
            "singleton_gene_fraction": float((genes["module_size"] == 1).mean()),
            "largest_module_fraction": float(module_sizes.max() / n_genes),
            "median_non_singleton_module_size": finite_quantile(
                module_sizes[module_sizes >= 2], 0.5
            ),
            "p25_non_singleton_module_size": finite_quantile(
                module_sizes[module_sizes >= 2], 0.25
            ),
            "p75_non_singleton_module_size": finite_quantile(
                module_sizes[module_sizes >= 2], 0.75
            ),
            "fraction_active_genes": float(active.mean()),
            "median_gene_margin": finite_quantile(genes["gene_margin"], 0.5),
            "p10_gene_margin": finite_quantile(genes["gene_margin"], 0.1),
            "fraction_negative_margin": float(
                (genes.loc[genes["gene_margin"].notna(), "gene_margin"] < 0).mean()
            ),
            "cross_edge_coverage": float((genes["cross_degree"] > 0).mean()),
            "reciprocal_gene_coverage": float(
                (genes["reciprocal_cross_fraction"].fillna(0) > 0).mean()
            ),
            "median_cross_strength_fraction": finite_quantile(
                genes["cross_strength_fraction"], 0.5
            ),
            "median_reciprocal_cross_fraction": finite_quantile(
                genes["reciprocal_cross_fraction"], 0.5
            ),
            "median_module_species_entropy": finite_quantile(entropy, 0.5),
            "gene_weighted_module_species_entropy": weighted_mean(
                entropy.reindex(modules["module"]).to_numpy(), modules["n_genes"]
            ),
            "fraction_genes_in_two_species_modules": float(
                genes["module"].isin(mixed_labels).mean()
            ),
        }

        cohesion_columns = [c for c in modules if c.startswith("cohesion_")]
        diversity_columns = [
            c for c in modules
            if c.startswith("diversity_") or c == "conserved_diversity"
        ]
        for column in cohesion_columns + diversity_columns:
            values = modules.loc[meaningful, column]
            row[f"median_{column}"] = finite_quantile(values, 0.5)
            if column == "cohesion_all":
                weights = modules.loc[meaningful, "n_genes"] * (
                    modules.loc[meaningful, "n_genes"] - 1
                )
            elif column.startswith("cohesion_"):
                suffix = column.removeprefix("cohesion_")
                species_count_columns = [
                    c for c in modules if c.startswith("n_") and c != "n_genes"
                ]
                species_names = [c.removeprefix("n_") for c in species_count_columns]
                match = None
                for left in species_names:
                    for right in species_names:
                        if suffix == f"{left}_{right}":
                            match = (left, right)
                            break
                    if match:
                        break
                if match is None:
                    weights = modules.loc[meaningful, "n_genes"]
                elif match[0] == match[1]:
                    counts = modules.loc[meaningful, f"n_{match[0]}"]
                    weights = counts * (counts - 1)
                else:
                    weights = (
                        modules.loc[meaningful, f"n_{match[0]}"]
                        * modules.loc[meaningful, f"n_{match[1]}"]
                    )
            else:
                weights = modules.loc[meaningful, "n_genes"]
            row[f"weighted_mean_{column}"] = weighted_mean(values, weights)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("algorithm")
    same_gene_universe = all(g == gene_universes[0] for g in gene_universes[1:])
    summary.insert(0, "same_gene_universe", same_gene_universe)

    pairwise_rows = []
    match_rows = []
    overlap_matrices = {}

    def add_directional_matches(source_name, target_name, joined):
        source_labels = joined[f"module_{source_name}"]
        target_labels = joined[f"module_{target_name}"]
        source_sizes = source_labels.value_counts()
        target_sizes = target_labels.value_counts()
        contingency = pd.crosstab(source_labels, target_labels)
        valid_source = source_sizes[source_sizes >= min_module_size].index
        valid_target = target_sizes[target_sizes >= min_module_size].index
        if len(valid_target) == 0:
            return
        for source_module in valid_source:
            intersections = contingency.loc[source_module].reindex(
                valid_target, fill_value=0
            )
            unions = (
                source_sizes[source_module]
                + target_sizes.reindex(valid_target)
                - intersections
            )
            jaccard = intersections / unions
            best_target = jaccard.idxmax()
            intersection = int(intersections[best_target])
            match_rows.append({
                "source_algorithm": source_name,
                "target_algorithm": target_name,
                "source_module": source_module,
                "target_module": best_target,
                "source_size": int(source_sizes[source_module]),
                "target_size": int(target_sizes[best_target]),
                "intersection": intersection,
                "jaccard": float(jaccard[best_target]),
                "source_recall": intersection / source_sizes[source_module],
                "target_recall": intersection / target_sizes[best_target],
            })

    clustering_pairs = combinations(prepared, 2) if include_pairwise else []
    for first_name, second_name in clustering_pairs:
        first = prepared[first_name]["genes"]
        second = prepared[second_name]["genes"]
        common = first.index.intersection(second.index)
        if len(common) < 2:
            raise ValueError(
                f"{first_name!r} and {second_name!r} share fewer than two genes"
            )
        joined = pd.DataFrame(index=common)
        for name, frame in ((first_name, first), (second_name, second)):
            joined[f"module_{name}"] = frame.loc[common, "module"].to_numpy()
            joined[f"active_{name}"] = frame.loc[common, "degree"].to_numpy() > 0
            joined[f"size_{name}"] = frame.loc[common, "module_size"].to_numpy()
            joined[f"margin_{name}"] = frame.loc[common, "gene_margin"].to_numpy()

        a = joined[f"module_{first_name}"].to_numpy()
        b = joined[f"module_{second_name}"].to_numpy()
        overlap_counts = pd.crosstab(
            joined[f"module_{first_name}"],
            joined[f"module_{second_name}"],
        )
        overlap_row_sizes = overlap_counts.sum(axis=1)
        overlap_col_sizes = overlap_counts.sum(axis=0)
        overlap_rows = overlap_row_sizes[
            overlap_row_sizes >= min_module_size
        ].sort_values(ascending=False).index
        overlap_cols = overlap_col_sizes[
            overlap_col_sizes >= min_module_size
        ].sort_values(ascending=False).index
        intersections = overlap_counts.loc[overlap_rows, overlap_cols]
        unions = (
            overlap_row_sizes.loc[overlap_rows].to_numpy()[:, None]
            + overlap_col_sizes.loc[overlap_cols].to_numpy()[None, :]
            - intersections.to_numpy()
        )
        overlap_matrices[(first_name, second_name)] = pd.DataFrame(
            np.divide(
                intersections.to_numpy(dtype=float), unions,
                out=np.zeros(intersections.shape, dtype=float),
                where=unions > 0,
            ),
            index=overlap_rows, columns=overlap_cols,
        )
        active_both = (
            joined[f"active_{first_name}"] & joined[f"active_{second_name}"]
        ).to_numpy()
        meaningful_both = (
            (joined[f"size_{first_name}"] >= min_module_size)
            & (joined[f"size_{second_name}"] >= min_module_size)
        ).to_numpy()

        def ari_on(mask):
            return adjusted_rand_score(a[mask], b[mask]) if mask.sum() >= 2 else np.nan

        pairwise_rows.append({
            "algorithm_a": first_name,
            "algorithm_b": second_name,
            "n_common_genes": len(common),
            "n_active_in_both": int(active_both.sum()),
            f"n_in_modules_ge_{min_module_size}_both": int(meaningful_both.sum()),
            "ari_all": adjusted_rand_score(a, b),
            "ari_active": ari_on(active_both),
            f"ari_modules_ge_{min_module_size}": ari_on(meaningful_both),
            "ami_all": adjusted_mutual_info_score(a, b),
            "nmi_all": normalized_mutual_info_score(a, b),
            "variation_of_information": variation_of_information(a, b),
            "margin_spearman": joined[
                [f"margin_{first_name}", f"margin_{second_name}"]
            ].corr(method="spearman").iloc[0, 1],
        })
        add_directional_matches(first_name, second_name, joined)
        add_directional_matches(second_name, first_name, joined)

    figures = {}
    if make_plots:
        from matplotlib.patches import Patch

        clustering_names = (
            list(prepared) if plot_clusterings is None else list(plot_clusterings)
        )
        unknown_plot_names = set(clustering_names) - set(prepared)
        if unknown_plot_names:
            raise ValueError(
                f"unknown plot clustering names {sorted(unknown_plot_names)}"
            )
        if not clustering_names:
            raise ValueError("plot_clusterings cannot be empty")
        clustering_colors = plt.get_cmap("tab10")(
            np.linspace(0, 1, max(len(clustering_names), 2))
        )
        metric_colors = ["#E1812C", "#3274A1", "#55A868"]
        all_species = pd.unique(pd.concat([
            item["genes"]["species"] for item in prepared.values()
        ], ignore_index=True))
        species_lookup = {str(value).lower(): value for value in all_species}
        if {"mouse", "human"}.issubset(species_lookup):
            mouse_name = species_lookup["mouse"]
            human_name = species_lookup["human"]
        else:
            mouse_name, human_name = all_species[:2]

        def draw_violin(axis, value_sets, colors, labels):
            valid = [(i + 1, np.asarray(values, dtype=float), color)
                     for i, (values, color) in enumerate(zip(value_sets, colors))
                     if np.isfinite(values).any()]
            if valid:
                positions = [item[0] for item in valid]
                arrays = [item[1][np.isfinite(item[1])] for item in valid]
                violins = axis.violinplot(
                    arrays, positions=positions, showmedians=True,
                    showextrema=False, widths=0.8,
                )
                for body, (_, _, color) in zip(violins["bodies"], valid):
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.65)
                violins["cmedians"].set_color("black")
            axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=25, ha="right")

        def draw_grouped_boxes(axis, column_specs, title, ylabel):
            offsets = np.linspace(-0.28, 0.28, len(column_specs))
            for offset, (column, label, color) in zip(offsets, column_specs):
                arrays, positions = [], []
                for index, name in enumerate(clustering_names):
                    modules = prepared[name]["modules"]
                    if column not in modules:
                        continue
                    values = modules.loc[
                        modules["n_genes"] >= min_module_size, column
                    ].dropna().to_numpy()
                    if len(values):
                        arrays.append(values)
                        positions.append(index + 1 + offset)
                if arrays:
                    boxes = axis.boxplot(
                        arrays, positions=positions, widths=0.22,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1.3},
                    )
                    for box in boxes["boxes"]:
                        box.set_facecolor(color)
                        box.set_alpha(0.65)
            axis.set_xticks(
                np.arange(1, len(clustering_names) + 1),
                clustering_names, rotation=25, ha="right",
            )
            axis.set(title=title, ylabel=ylabel)
            axis.legend(
                handles=[Patch(facecolor=color, alpha=0.65, label=label)
                         for _, label, color in column_specs],
                frameon=False, ncol=len(column_specs), loc="best",
            )

        # 1. Ranked module-size distributions.
        fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=130)
        for name, color in zip(clustering_names, clustering_colors):
            sizes = np.sort(prepared[name]["modules"]["n_genes"].to_numpy())[::-1]
            ax.plot(np.arange(1, len(sizes) + 1), sizes, lw=1.8, color=color,
                    label=f"{name} ({len(sizes)} modules)")
        ax.axhline(min_module_size, color="0.45", ls="--", lw=1,
                   label=f"minimum size = {min_module_size}")
        ax.set_yscale("log")
        ax.set(xlabel="Module rank", ylabel="Genes in module (log scale)",
               title="Ranked module-size distributions")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        figures["ranked_module_sizes"] = fig

        # 2. Gene-margin distributions across all genes.
        fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(clustering_names)), 4.8), dpi=130)
        margin_values = [prepared[name]["genes"]["gene_margin"].to_numpy()
                         for name in clustering_names]
        draw_violin(ax, margin_values, clustering_colors, clustering_names)
        ax.axhline(0, color="#B22222", ls="--", lw=1)
        ax.set(ylabel="Gene margin", title="Gene-margin distributions")
        fig.tight_layout()
        figures["gene_margin_violin"] = fig

        # 3. Cohesion distributions over sufficiently large modules.
        cohesion_specs = [
            (f"cohesion_{mouse_name}_{mouse_name}", str(mouse_name), metric_colors[0]),
            (f"cohesion_{human_name}_{human_name}", str(human_name), metric_colors[1]),
            (f"cohesion_{mouse_name}_{human_name}", "interspecies", metric_colors[2]),
        ]
        fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(clustering_names)), 5.0), dpi=130)
        draw_grouped_boxes(ax, cohesion_specs, "Module cohesion by clustering",
                           "Mean second-order correlation per module")
        fig.tight_layout()
        figures["module_cohesion_boxplot"] = fig

        # 4. Diversity has species-specific and conserved values; there is no
        # direct interspecies diversity because spatial coordinates are separate.
        diversity_specs = [
            (f"diversity_{mouse_name}", str(mouse_name), metric_colors[0]),
            (f"diversity_{human_name}", str(human_name), metric_colors[1]),
            ("conserved_diversity", "conserved (minimum)", metric_colors[2]),
        ]
        fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(clustering_names)), 5.0), dpi=130)
        draw_grouped_boxes(ax, diversity_specs, "Module diversity by clustering",
                           "Spatial diversity per module")
        fig.tight_layout()
        figures["module_diversity_boxplot"] = fig

        # 5. Cross-species strength fraction over genes.
        fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(clustering_names)), 4.8), dpi=130)
        cross_values = [prepared[name]["genes"]["cross_strength_fraction"].to_numpy()
                        for name in clustering_names]
        draw_violin(ax, cross_values, clustering_colors, clustering_names)
        ax.set(ylabel="Cross-species strength fraction",
               title="Cross-species integration strength", ylim=(-0.02, 1.02))
        fig.tight_layout()
        figures["cross_strength_violin"] = fig

        # 6. A clustering-level ARI heatmap summarizes pairwise agreement.
        if pairwise_rows:
            ari_matrix = pd.DataFrame(
                np.eye(len(clustering_names)), index=clustering_names, columns=clustering_names
            )
            for row in pairwise_rows:
                if (row["algorithm_a"] in ari_matrix.index
                        and row["algorithm_b"] in ari_matrix.index):
                    ari_matrix.loc[row["algorithm_a"], row["algorithm_b"]] = row["ari_all"]
                    ari_matrix.loc[row["algorithm_b"], row["algorithm_a"]] = row["ari_all"]
            fig, ax = plt.subplots(figsize=(max(5.5, 0.8 * len(clustering_names) + 2.5),
                                            max(4.8, 0.7 * len(clustering_names) + 2.0)), dpi=130)
            image = ax.imshow(ari_matrix, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(clustering_names)), clustering_names, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(clustering_names)), clustering_names)
            for row in range(len(clustering_names)):
                for column in range(len(clustering_names)):
                    value = ari_matrix.iat[row, column]
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                            color="white" if value < 0.55 else "black")
            fig.colorbar(image, ax=ax, label="Adjusted Rand index")
            ax.set_title("Overall partition agreement")
            fig.tight_layout()
            figures["ari_heatmap"] = fig

        # 7. Full module-overlap heatmaps compare each setting with the reference.
        for pair, overlap in overlap_matrices.items():
            if reference not in pair or not set(pair).issubset(clustering_names):
                continue
            first_name, second_name = pair
            width = min(14, max(6, 0.30 * max(len(overlap.columns), 1) + 3))
            height = min(14, max(5, 0.30 * max(len(overlap.index), 1) + 2))
            fig, ax = plt.subplots(figsize=(width, height), dpi=130)
            if overlap.empty:
                ax.text(0.5, 0.5, f"No modules contain at least {min_module_size} common genes",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                image = ax.imshow(overlap, aspect="auto", cmap="magma",
                                  vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks(np.arange(len(overlap.columns)),
                              [str(value) for value in overlap.columns], rotation=90, fontsize=7)
                ax.set_yticks(np.arange(len(overlap.index)),
                              [str(value) for value in overlap.index], fontsize=7)
                fig.colorbar(image, ax=ax, label="Module Jaccard overlap")
                ax.set(xlabel=f"{second_name} module", ylabel=f"{first_name} module")
            ax.set_title(f"Module overlap: {first_name} versus {second_name}")
            fig.tight_layout()
            figures[f"module_overlap__{first_name}__{second_name}"] = fig

        if show_plots:
            plt.show()

    return {
        "summary": summary,
        "pairwise": pd.DataFrame(pairwise_rows),
        "module_matches": pd.DataFrame(match_rows),
        "overlap_matrices": overlap_matrices,
        "figures": figures,
    }



def evaluate_clusterings(
    clusterings,
    C,
    prof,
    *,
    expression_by_species=None,
    min_module_size=10,
    gene_margin_min_module_size=2,
    diversity_min_genes=2,
    reference=None,
    make_plots=True,
    show_plots=True,
    include_pairwise=True,
    plot_clusterings=None,
    max_clusterings=10,
    output_dir=None,
    plot_format="png",
    return_figures=True,
):
    """Evaluate several completed clusterings and compare their outputs.

    ``clusterings`` is a list of dictionaries. Each dictionary requires
    ``name``, ``G``, and ``labels`` and may include ``k_inter`` and
    ``cutoff_inter`` when those values are not stored in ``G.meta``.

    ``max_clusterings`` is a safety limit against accidentally retaining too
    many graphs and evaluations. When ``output_dir`` is provided, all numeric
    outputs and figures are saved there. Set ``show_plots=False`` and
    ``return_figures=False`` to write figures without displaying or retaining
    them. Individual gene tables use Parquet when available and fall back to
    compressed CSV.
    """
    if not isinstance(clusterings, (list, tuple)) or len(clusterings) < 2:
        raise ValueError("clusterings must be a list containing at least two entries")
    if not isinstance(max_clusterings, int) or max_clusterings < 2:
        raise ValueError("max_clusterings must be an integer of at least 2")
    if len(clusterings) > max_clusterings:
        raise ValueError(
            f"received {len(clusterings)} clusterings; the safety limit is "
            f"{max_clusterings}. Pass smaller batches rather than retaining "
            "all graphs and evaluations in memory."
        )
    plot_format = str(plot_format).lower().lstrip(".")
    if plot_format not in {"png", "pdf", "svg"}:
        raise ValueError("plot_format must be 'png', 'pdf', or 'svg'")

    from pathlib import Path

    output_root = Path(output_dir) if output_dir is not None else None
    if output_root is not None:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)

    def safe_filename(value):
        text = str(value).strip()
        cleaned = "".join(
            character if character.isalnum() or character in {"-", "_"}
            else "_" for character in text
        ).strip("_")
        return cleaned or "clustering"

    def write_frame(frame, base_path):
        try:
            path = base_path.with_suffix(".parquet")
            frame.to_parquet(path, index=False)
        except (ImportError, ModuleNotFoundError):
            path = base_path.with_suffix(".csv.gz")
            frame.to_csv(path, index=False, compression="gzip")
        return path

    evaluations = {}
    setting_rows = []
    labels_by_name = {}
    for index, specification in enumerate(clusterings):
        if not isinstance(specification, dict):
            raise TypeError(f"clustering {index} must be a dictionary")
        missing = {"name", "G", "labels"} - set(specification)
        if missing:
            raise ValueError(f"clustering {index} is missing {sorted(missing)}")
        name = str(specification["name"])
        if name in evaluations:
            raise ValueError(f"duplicate clustering name {name!r}")
        graph = specification["G"]
        k_inter = specification.get("k_inter", graph.meta.get("k_inter"))
        cutoff_inter = specification.get(
            "cutoff_inter", graph.meta.get("cutoff_inter")
        )
        evaluations[name] = evaluate_joint_modules(
            specification["labels"], C, graph, prof,
            expression_by_species=expression_by_species,
            k_inter=k_inter, cutoff_inter=cutoff_inter,
            min_module_size=gene_margin_min_module_size,
            diversity_min_genes=diversity_min_genes,
            make_plots=False,
        )
        labels_by_name[name] = np.asarray(specification["labels"])
        setting_rows.append({
            key: value for key, value in specification.items()
            if key not in {"G", "labels"}
        })

    if reference is None:
        reference = next(iter(evaluations))
    comparison = compare_clustering_evaluations(
        evaluations, min_module_size=min_module_size,
        make_plots=make_plots, show_plots=show_plots, reference=reference,
        include_pairwise=include_pairwise,
        plot_clusterings=plot_clusterings,
    )
    settings = pd.DataFrame(setting_rows).set_index("name")
    figure_paths = {}
    manifest_rows = []
    if output_root is not None:
        settings.to_csv(output_root / "settings.csv")
        comparison["summary"].to_csv(output_root / "summary.csv")
        comparison["pairwise"].to_csv(output_root / "pairwise.csv", index=False)
        comparison["module_matches"].to_csv(
            output_root / "module_matches.csv", index=False
        )
        overlap_dir = output_root / "overlap_matrices"
        if comparison["overlap_matrices"]:
            overlap_dir.mkdir(exist_ok=True)
            for pair, overlap in comparison["overlap_matrices"].items():
                overlap.to_csv(
                    overlap_dir / f"{safe_filename(pair[0])}__{safe_filename(pair[1])}.csv"
                )

        individual_dir = output_root / "individual_evaluations"
        individual_dir.mkdir(exist_ok=True)
        for index, (name, evaluation) in enumerate(evaluations.items()):
            run_dir = individual_dir / f"{index:02d}_{safe_filename(name)}"
            run_dir.mkdir()
            np.save(run_dir / "labels.npy", labels_by_name[name])
            gene_path = write_frame(evaluation["genes"], run_dir / "genes")
            module_path = write_frame(evaluation["modules"], run_dir / "modules")
            evaluation["integration"].to_csv(run_dir / "integration.csv", index=False)
            evaluation["summary"].to_frame("value").to_csv(run_dir / "summary.csv")
            manifest_rows.append({
                "name": name, "directory": str(run_dir),
                "genes_file": str(gene_path), "modules_file": str(module_path),
                "labels_file": str(run_dir / "labels.npy"),
            })

        plots_dir = output_root / "plots"
        if comparison["figures"]:
            plots_dir.mkdir(exist_ok=True)
            for name, figure in comparison["figures"].items():
                path = plots_dir / f"{safe_filename(name)}.{plot_format}"
                figure.savefig(path, dpi=250, bbox_inches="tight")
                figure_paths[name] = str(path)
        pd.DataFrame(manifest_rows).to_csv(output_root / "manifest.csv", index=False)

    if not return_figures:
        for figure in comparison["figures"].values():
            plt.close(figure)
        comparison["figures"] = {}

    return {
        "evaluations": evaluations,
        "settings": settings,
        "manifest": pd.DataFrame(manifest_rows),
        "figure_paths": figure_paths,
        "output_dir": str(output_root) if output_root is not None else None,
        **comparison,
    }
