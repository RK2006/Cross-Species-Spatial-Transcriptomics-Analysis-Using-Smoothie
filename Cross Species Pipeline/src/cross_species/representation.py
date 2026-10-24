"""Stitched ortholog profiles and second-order gene similarities."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd



@dataclass
class StitchedProfiles:
    z: np.ndarray                 # (n_ortho, n_genes) Fisher-z profiles
    genes: np.ndarray             # species-tagged gene names
    species: np.ndarray           # species label per gene
    raw_genes: np.ndarray         # original untagged names
    meta: dict = field(default_factory=dict)

    @property
    def features(self) -> np.ndarray:
        """(n_genes, n_ortho) — genes as observations, NaNs zeroed."""
        return np.nan_to_num(self.z, nan=0.0).T.astype(np.float32)

    def __repr__(self) -> str:
        a, b = self.meta["name_a"], self.meta["name_b"]
        na = int((self.species == a).sum())
        nb = int((self.species == b).sum())
        return (f"<StitchedProfiles {self.z.shape[0]} orthologs x "
                f"{self.z.shape[1]:,} genes ({a} {na:,} | {b} {nb:,})>")


def stitch_profiles(
    res=None,
    n_ortho: int | None = None,
    *,
    mat_a=None, genes_a=None, mat_b=None, genes_b=None,
    name_a: str | None = None,
    name_b: str | None = None,
    clip: float = 0.9999,
    mask_self: bool = True,
    verbose: bool = True,
) -> StitchedProfiles:
    """Stitch A[:X, :] and B[:X, :] side by side, tag gene names, Fisher-z.

    Result is (n_ortho, n_a + n_b): each column a gene from either species, each
    row one of the X shared orthologs. Gene names become "<species>:<symbol>".

    mask_self
        Blank each ortholog gene's self-correlation. It sits at 1.0, becomes the
        largest value in that gene's profile after arctanh, and exists only for
        genes in the leading block — an artefact that would otherwise separate
        the ortholog block from the tail for pure bookkeeping reasons.
    """
    if res is not None:
        mat_a, genes_a = res.matrix_a, res.genes_a
        mat_b, genes_b = res.matrix_b, res.genes_b
        name_a = name_a or str(res.meta.get("species_a", "a")).split(",")[0].strip()
        name_b = name_b or str(res.meta.get("species_b", "b")).split(",")[0].strip()
        n_ortho = n_ortho or res.n_orthologs
    name_a, name_b = name_a or "a", name_b or "b"
    if n_ortho is None:
        raise ValueError("n_ortho is required when res is not given")

    A = np.asarray(mat_a)[:n_ortho, :]
    B = np.asarray(mat_b)[:n_ortho, :]
    if A.shape[0] != n_ortho or B.shape[0] != n_ortho:
        raise ValueError(f"need {n_ortho} ortholog rows; got {A.shape[0]}, {B.shape[0]}")
    if len(genes_a) != A.shape[1] or len(genes_b) != B.shape[1]:
        raise ValueError("gene name lists do not match matrix widths")

    wa = A.shape[1]
    dtype = np.float32 if (A.dtype == np.float32 and B.dtype == np.float32) else np.float64
    z = np.empty((n_ortho, wa + B.shape[1]), dtype=dtype)
    z[:, :wa] = A
    z[:, wa:] = B

    np.clip(z, -clip, clip, out=z)
    np.arctanh(z, out=z)

    if mask_self:
        ka = min(n_ortho, wa)
        z[np.arange(ka), np.arange(ka)] = np.nan
        kb = min(n_ortho, z.shape[1] - wa)
        z[np.arange(kb), wa + np.arange(kb)] = np.nan

    raw = np.asarray(list(genes_a) + list(genes_b))
    genes = np.asarray([f"{name_a}:{g}" for g in genes_a]
                       + [f"{name_b}:{g}" for g in genes_b])
    species = np.asarray([name_a] * wa + [name_b] * B.shape[1])

    prof = StitchedProfiles(
        z=z, genes=genes, species=species, raw_genes=raw,
        meta=dict(n_ortho=n_ortho, width_a=wa, name_a=name_a, name_b=name_b,
                  clip=clip, mask_self=mask_self),
    )
    if verbose:
        print(prof)
    return prof



@dataclass
class EdgeSets:
    bins: np.ndarray
    counts: dict = field(default_factory=dict)
    stats: pd.DataFrame = None
    meta: dict = field(default_factory=dict)

    def density(self, label: str) -> np.ndarray:
        counts = self.counts[label].astype(float)
        return counts / max(counts.sum(), 1)


# Shared figure conventions for the representation and search diagnostics.
ANALYSIS_PLOT_FORMAT = "png"
ANALYSIS_FIGURE_DPI = 130
ANALYSIS_SAVE_DPI = 250
ANALYSIS_WIDE_FIGSIZE = (13.5, 4.2)
ANALYSIS_GRID_ALPHA = 0.18
REPRESENTATION_COLORS = {
    "no_pca": "#222222",
    "pca_050": "#2C7FB8",
    "pca_100": "#31A354",
}


def style_analysis_axes(axes):
    """Apply one visual style to all representation/search figures."""
    import numpy as np

    for axis in np.asarray(axes, dtype=object).ravel():
        if axis.get_visible():
            axis.grid(alpha=ANALYSIS_GRID_ALPHA)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)


def save_analysis_figures(figures, output_dir):
    """Save a named figure mapping with the shared format and resolution."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, figure in figures.items():
        path = output_dir / f"{name}.{ANALYSIS_PLOT_FORMAT}"
        figure.savefig(path, dpi=ANALYSIS_SAVE_DPI, bbox_inches="tight")
        paths[name] = str(path)
    return paths


def pca_pearson_embedding(pca_scores, dimensions, dtype=np.float32):
    """Prepare PCA-score rows so their dot products equal Pearson correlation.

    Each gene is represented by its first ``dimensions`` PCA scores. Centering
    each gene across those retained components and then scaling it to unit
    Euclidean length turns a row-wise dot product into the Pearson correlation
    between the two genes' PCA-score vectors. Zero-variance rows remain zero.
    """
    scores = np.asarray(pca_scores)
    dimensions = int(dimensions)
    if scores.ndim != 2 or not 2 <= dimensions <= scores.shape[1]:
        raise ValueError(
            "dimensions must be between 2 and the width of the PCA score matrix"
        )
    embedding = np.asarray(scores[:, :dimensions], dtype=dtype).copy()
    embedding -= embedding.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding /= np.where(norms > 0, norms, 1).astype(dtype)
    return embedding


def similarity_from_unit_rows(
    rows,
    *,
    return_matrix=True,
    collect_edges=True,
    split=None,
    name_a="A",
    name_b="B",
    bins=200,
    dtype=np.float32,
    block=2048,
    meta=None,
):
    """Compute dot-product similarities and optional pair distributions once.

    This is the shared implementation used by ``profile_correlation``, the PCA
    pilot, the cutoff-calibration plots, and the Sobol search. ``rows`` must
    already be preprocessed so their dot products are the intended similarity.
    Centered, unit-length rows therefore produce Pearson correlations.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or block < 1:
        raise ValueError("rows must be two-dimensional and block must be positive")
    n = rows.shape[0]
    if collect_edges and (split is None or not 0 < int(split) < n):
        raise ValueError("an interior split is required when collect_edges=True")
    split = int(split) if split is not None else None
    edges = (
        np.linspace(-1.0, 1.0, int(bins) + 1)
        if np.isscalar(bins) else np.asarray(bins, dtype=float)
    )
    if collect_edges and (
        edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("bins must be a positive count or increasing edge array")

    labels = (f"{name_a}-{name_a}", f"{name_b}-{name_b}", f"{name_a}-{name_b}")
    counts = {
        label: np.zeros(len(edges) - 1, dtype=np.int64) for label in labels
    } if collect_edges else None
    accumulators = {label: [0, 0.0, 0.0] for label in labels} if collect_edges else None
    C = np.empty((n, n), dtype=dtype) if return_matrix else None

    for start in range(0, n, block):
        stop = min(start + block, n)
        similarity = rows[start:stop] @ rows.T
        np.clip(similarity, -1.0, 1.0, out=similarity)
        if C is not None:
            C[start:stop] = similarity.astype(dtype, copy=False)
        if not collect_edges:
            continue

        row_indices = np.arange(start, stop)
        row_a = row_indices < split
        for column_start, column_stop, column_a in (
            (0, split, True), (split, n, False)
        ):
            submatrix = similarity[:, column_start:column_stop]
            for is_a, mask in ((True, row_a), (False, ~row_a)):
                if not mask.any():
                    continue
                selected = submatrix[mask]
                selected_rows = row_indices[mask]
                if is_a == column_a:
                    keep = selected_rows[:, None] < np.arange(
                        column_start, column_stop
                    )[None, :]
                    values = selected[keep]
                    label = labels[0] if is_a else labels[1]
                elif is_a and not column_a:
                    values = selected.ravel()
                    label = labels[2]
                else:
                    continue
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                counts[label] += np.histogram(values, bins=edges)[0]
                accumulators[label][0] += values.size
                accumulators[label][1] += float(values.sum())
                accumulators[label][2] += float(np.square(values).sum())

    if C is not None:
        np.fill_diagonal(C, 1.0)
    if not collect_edges:
        return C, None

    centers = (edges[:-1] + edges[1:]) / 2
    stats_rows = []
    for label, (count, value_sum, square_sum) in accumulators.items():
        if count == 0:
            continue
        mean = value_sum / count
        variance = max(square_sum / count - mean ** 2, 0.0)
        histogram = counts[label]
        cdf = np.cumsum(histogram) / histogram.sum()
        stats_rows.append({
            "edge_set": label,
            "kind": "within" if label in labels[:2] else "cross",
            "n_edges": count,
            "mean": mean,
            "sd": np.sqrt(variance),
            "median": centers[np.searchsorted(cdf, 0.5)],
            "frac_gt_0_3": float(histogram[centers > 0.3].sum() / histogram.sum()),
            "frac_gt_0_5": float(histogram[centers > 0.5].sum() / histogram.sum()),
        })
    stats = pd.DataFrame(stats_rows)
    within = stats.loc[stats.kind == "within", "mean"].mean()
    cross = stats.loc[stats.kind == "cross", "mean"].mean()
    stats.attrs["floor_offset"] = within - cross
    edge_meta = {
        "n": n,
        "width_a": split,
        "name_a": str(name_a),
        "name_b": str(name_b),
        **({} if meta is None else dict(meta)),
    }
    return C, EdgeSets(bins=edges, counts=counts, stats=stats, meta=edge_meta)


def profile_correlation(
    prof,
    return_matrix: bool = True,
    dtype=np.float32,
    bins: int = 200,
    block: int = 2048,
    max_dense: int = 30000,
    use_pca: bool = False,
    n_components: int = 200,
    verbose: bool = True,
) -> tuple[np.ndarray | None, EdgeSets]:
    """Pairwise similarity between gene profiles across shared orthologs.

    Without PCA this returns the original Pearson correlations across ortholog
    features. With PCA it returns Pearson correlations between each pair of
    genes' retained PCA-score vectors. Matrix construction and pair-type
    summaries share one blockwise helper.
    """
    X = prof.features.astype(np.float64)
    n, original_features = X.shape
    split = int(prof.meta["width_a"])
    name_a, name_b = prof.meta["name_a"], prof.meta["name_b"]
    if n > max_dense and return_matrix:
        raise MemoryError(
            f"{n:,} genes -> {n*n*np.dtype(dtype).itemsize/1e9:.1f} GB matrix; "
            "raise max_dense or pass return_matrix=False"
        )
    if not isinstance(use_pca, (bool, np.bool_)):
        raise TypeError("use_pca must be True or False")
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if use_pca and n_components < 2:
        raise ValueError("PCA-score Pearson correlation requires at least 2 components")
    if use_pca and n_components > min(n, original_features):
        raise ValueError(
            f"n_components={n_components} exceeds the maximum "
            f"{min(n, original_features)} for an {n} x {original_features} matrix"
        )

    X -= X.mean(axis=1, keepdims=True)
    standard_deviation = X.std(axis=1, keepdims=True)
    X /= np.where(standard_deviation > 0, standard_deviation, 1)

    explained_variance = None
    if use_pca:
        from sklearn.decomposition import PCA

        if verbose:
            print(
                f"fitting joint PCA: {original_features:,} profile features "
                f"-> {n_components:,} components"
            )
        pca = PCA(
            n_components=int(n_components), svd_solver="randomized",
            whiten=False, random_state=0,
        )
        representation = pca_pearson_embedding(
            pca.fit_transform(X), n_components, dtype=np.float64
        )
        explained_variance = float(pca.explained_variance_ratio_.sum())
        feature_label = "PCA components"
        similarity_name = "Pearson correlation between PCA-score vectors"
        if verbose:
            print(f"PCA cumulative explained variance: {explained_variance:.1%}")
    else:
        representation = X / np.sqrt(original_features)
        feature_label = "ortholog features"
        similarity_name = "Pearson correlation between gene profiles"

    if verbose:
        print(
            f"correlating {n:,} gene profiles over {representation.shape[1]:,} "
            f"{feature_label}"
            + (f" -> {n*n*np.dtype(dtype).itemsize/1e9:.2f} GB matrix" if return_matrix else "")
        )
    C, edge_sets = similarity_from_unit_rows(
        representation,
        return_matrix=return_matrix,
        collect_edges=True,
        split=split,
        name_a=name_a,
        name_b=name_b,
        bins=bins,
        dtype=dtype,
        block=block,
        meta={
            "use_pca": bool(use_pca),
            "n_components": representation.shape[1] if use_pca else None,
            "original_features": original_features,
            "explained_variance": explained_variance,
            "similarity_name": similarity_name,
        },
    )
    if verbose:
        print(edge_sets.stats.to_string(index=False))
        print(
            "\nwithin-species mean minus cross-species mean = "
            f"{edge_sets.stats.attrs['floor_offset']:+.4f}"
        )
    return C, edge_sets


def plot_edges(edges: EdgeSets, log: bool = True, cumulative: bool = False):
    """Plot the three pair distributions and their within-minus-cross difference."""
    import matplotlib.pyplot as plt

    name_a, name_b = edges.meta["name_a"], edges.meta["name_b"]
    centers = (edges.bins[:-1] + edges.bins[1:]) / 2
    colors = {
        f"{name_a}-{name_a}": "#E1812C",
        f"{name_b}-{name_b}": "#3274A1",
        f"{name_a}-{name_b}": "#3A923A",
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 3.8), dpi=ANALYSIS_FIGURE_DPI,
        constrained_layout=True,
    )
    for label, color in colors.items():
        density = edges.density(label)
        values = np.cumsum(density) if cumulative else density
        axes[0].plot(
            centers, values, color=color, lw=1.6,
            label=f"{label} (n={edges.counts[label].sum():,})",
        )
        axes[0].fill_between(centers, values, color=color, alpha=0.14)
    if log and not cumulative:
        axes[0].set_yscale("log")
    axes[0].axvline(0, ls=":", lw=0.8, color="0.5")
    similarity_name = edges.meta.get(
        "similarity_name", "Similarity between gene profiles"
    )
    axes[0].set(
        xlabel=similarity_name,
        ylabel="Cumulative fraction" if cumulative else "Pair fraction",
        title="Pairwise similarity distributions",
    )
    axes[0].legend(fontsize=7, frameon=False)

    within = (
        edges.density(f"{name_a}-{name_a}")
        + edges.density(f"{name_b}-{name_b}")
    ) / 2
    axes[1].plot(
        centers, within - edges.density(f"{name_a}-{name_b}"),
        color="0.25", lw=1.6,
    )
    axes[1].axhline(0, ls=":", lw=0.8, color="0.5")
    axes[1].axvline(0, ls=":", lw=0.8, color="0.5")
    axes[1].set(
        xlabel=similarity_name,
        ylabel="Within minus cross density",
        title="Excess within-species density",
    )
    style_analysis_axes(axes)
    return figure
