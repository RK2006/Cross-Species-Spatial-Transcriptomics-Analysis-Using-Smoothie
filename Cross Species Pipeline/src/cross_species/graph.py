"""
crossspecies_graph.py — build a gene graph with separate within- and
cross-dataset edge budgets.

    G = dual_knn_graph(C, split=prof.meta["width_a"], k_intra=15, k_inter=5)
    G.summary()
    labels = G.leiden(resolution=1.0)

Each gene proposes its top `k_intra` partners from its own dataset and its top
`k_inter` from the other, then proposals are symmetrized. Separating the budgets
is what lets the cross-species edges exist at all: a plain top-k graph on these
profiles fills every slot with same-species partners, because within-dataset
correlations have far heavier tails than cross-dataset ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class DualGraph:
    adj: sparse.csr_matrix              # symmetric, weights = Pearson r
    species: np.ndarray                 # per-node dataset label
    genes: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.adj.shape[0]

    def edge_frame(self) -> pd.DataFrame:
        """Upper-triangle edge list with a within/cross tag."""
        A = sparse.triu(self.adj, k=1).tocoo()
        same = self.species[A.row] == self.species[A.col]
        df = pd.DataFrame({
            "i": A.row, "j": A.col, "weight": A.data,
            "kind": np.where(same, "within", "cross"),
        })
        if self.genes is not None:
            df["gene_i"] = self.genes[A.row]
            df["gene_j"] = self.genes[A.col]
        return df

    def to_csv(
        self,
        path: str,
        nodes_path: str | None = None,
        labels: np.ndarray | None = None,
        float_format: str = "%.6g",
        index_cols: bool = False,
        verbose: bool = True,
    ) -> str:
        """Write the edge list to CSV (one row per undirected edge).

        Columns: source, target, weight, kind — plus source_species /
        target_species, and cluster columns when `labels` is given. Cytoscape,
        Gephi, and igraph's read_ncol all accept this layout.

        A path ending in .gz / .zst is compressed automatically by pandas.
        `index_cols=True` keeps the integer node indices alongside the names,
        which is what you want if you plan to read the graph back and line it up
        against the correlation matrix.
        """
        df = self.edge_frame()
        if self.genes is not None:
            out = pd.DataFrame({
                "source": df.gene_i, "target": df.gene_j,
                "weight": df.weight, "kind": df.kind,
                "source_species": self.species[df.i],
                "target_species": self.species[df.j],
            })
        else:
            out = pd.DataFrame({
                "source": df.i, "target": df.j,
                "weight": df.weight, "kind": df.kind,
                "source_species": self.species[df.i],
                "target_species": self.species[df.j],
            })
        if index_cols:
            out.insert(2, "source_idx", df.i.values)
            out.insert(3, "target_idx", df.j.values)
        if labels is not None:
            labels = np.asarray(labels)
            out["source_cluster"] = labels[df.i]
            out["target_cluster"] = labels[df.j]
            out["same_cluster"] = out.source_cluster == out.target_cluster

        out.to_csv(path, index=False, float_format=float_format)
        written = [path]

        if nodes_path is not None:
            nodes = self.degrees()
            if labels is not None:
                nodes["cluster"] = labels
            nodes.insert(0, "node_idx", np.arange(self.n_nodes))
            nodes.to_csv(nodes_path, index=False, float_format=float_format)
            written.append(nodes_path)

        if verbose:
            n_cross = int((out.kind == "cross").sum())
            print(f"wrote {len(out):,} edges ({n_cross:,} cross) -> {path}"
                  + (f"\n      {self.n_nodes:,} nodes -> {nodes_path}"
                     if nodes_path else ""))
        return path

    def degrees(self) -> pd.DataFrame:
        """Per-node within / cross degree."""
        A = self.adj.tocoo()
        same = self.species[A.row] == self.species[A.col]
        within = np.bincount(A.row[same], minlength=self.n_nodes)
        cross = np.bincount(A.row[~same], minlength=self.n_nodes)
        return pd.DataFrame({
            "gene": self.genes if self.genes is not None else np.arange(self.n_nodes),
            "species": self.species,
            "deg_within": within, "deg_cross": cross,
            "deg_total": within + cross,
            "frac_cross": cross / np.maximum(within + cross, 1),
        })

    def summary(self) -> None:
        d = self.degrees()
        A = sparse.triu(self.adj, k=1).tocoo()
        same = self.species[A.row] == self.species[A.col]
        print(f"{self.n_nodes:,} nodes | {A.nnz:,} edges "
              f"({same.sum():,} within, {(~same).sum():,} cross)")
        print(f"  isolated nodes        : {(d.deg_total == 0).sum():,}")
        print(f"  nodes with 0 cross    : {(d.deg_cross == 0).sum():,} "
              f"({(d.deg_cross == 0).mean():.1%})")
        print(f"  median degree         : within {d.deg_within.median():.0f}, "
              f"cross {d.deg_cross.median():.0f}")
        if A.nnz:
            print(f"  mean weight           : within {A.data[same].mean():.3f}, "
                  f"cross {A.data[~same].mean():.3f}")
        sat = self.meta.get("saturated")
        if sat:
            for tag in ("intra", "inter"):
                k = self.meta.get(f"k_{tag}")
                cut = self.meta.get(f"cutoff_{tag}")
                if k is None:
                    print(f"  {tag:<5} constraint     : cutoff {cut} only")
                else:
                    print(f"  {tag:<5} constraint     : k={k}"
                          + (f", cutoff {cut} — k binding for "
                             f"{sat[tag]:,}/{self.n_nodes:,} genes" if cut is not None
                             else " (no cutoff)"))
        n_comp, lab = sparse.csgraph.connected_components(self.adj, directed=False)
        sizes = np.bincount(lab)
        print(f"  connected components  : {n_comp:,} (largest {sizes.max():,})")

    def to_igraph(self, weight_mode: str = "raw", cross_boost: float = 1.0):
        """Build the igraph object, optionally rebalancing edge weights.

        weight_mode
            "raw"   weights are the Pearson r as stored.
            "rank"  weights are replaced by within-class percentile ranks, so
                    within- and cross-dataset edges are placed on a common
                    scale. Use this when cross edges are real but numerically
                    smaller and so never influence community structure.
            "boost" raw weights with cross-dataset edges multiplied by
                    `cross_boost`.
        """
        import igraph as ig

        A = sparse.triu(self.adj, k=1).tocoo()
        w = A.data.astype(float).copy()
        cross = self.species[A.row] != self.species[A.col]

        if weight_mode == "rank":
            for m in (cross, ~cross):
                if m.sum() > 1:
                    r = np.argsort(np.argsort(w[m]))
                    w[m] = (r + 1) / (m.sum() + 1)
        elif weight_mode == "boost":
            w[cross] *= cross_boost
        elif weight_mode != "raw":
            raise ValueError(f"unknown weight_mode {weight_mode!r}")

        g = ig.Graph(n=self.n_nodes, edges=list(zip(A.row.tolist(), A.col.tolist())))
        g.es["weight"] = w.tolist()
        g.vs["species"] = list(self.species)
        if self.genes is not None:
            g.vs["name"] = list(self.genes)
        return g

    def community(
        self,
        method: str = "leiden",
        resolution: float = 1.0,
        weight_mode: str = "raw",
        cross_boost: float = 1.0,
        objective: str = "modularity",
        n_iterations: int = -1,
        seed: int = 0,
        verbose: bool = True,
    ) -> np.ndarray:
        """Community detection using igraph's own algorithms (no leidenalg).

        method
            leiden, louvain (multilevel), infomap, walktrap, fastgreedy,
            leading_eigenvector, label_propagation
        """
        import igraph as ig
        import random

        random.seed(seed)
        g = self.to_igraph(weight_mode=weight_mode, cross_boost=cross_boost)
        w = g.es["weight"]

        if method == "leiden":
            vc = g.community_leiden(objective_function=objective, weights=w,
                                    resolution=resolution, n_iterations=n_iterations)
        elif method in ("louvain", "multilevel"):
            vc = g.community_multilevel(weights=w, resolution=resolution)
        elif method == "infomap":
            vc = g.community_infomap(edge_weights=w)
        elif method == "walktrap":
            vc = g.community_walktrap(weights=w).as_clustering()
        elif method == "fastgreedy":
            vc = g.community_fastgreedy(weights=w).as_clustering()
        elif method == "leading_eigenvector":
            vc = g.community_leading_eigenvector(weights=w)
        elif method == "label_propagation":
            vc = g.community_label_propagation(weights=w)
        else:
            raise ValueError(f"unknown method {method!r}")

        labels = np.asarray(vc.membership)
        if verbose:
            sizes = np.bincount(labels)
            mix = self.mixing(labels)
            print(f"{method}: {len(sizes)} communities, "
                  f"largest {sizes.max():,}, singletons {(sizes == 1).sum():,}")
            print(f"  modularity {g.modularity(labels, weights=w):.4f} | "
                  f"mean frac_minor {mix.frac_minor.mean():.3f} | "
                  f"mixed communities {(mix.frac_minor > 0.05).sum()}/{len(mix)}")
        return labels

    def compare_methods(self, methods=None, **kw) -> pd.DataFrame:
        """Run several igraph community algorithms and tabulate the outcomes."""
        methods = methods or ["leiden", "louvain", "infomap", "walktrap",
                              "fastgreedy", "leading_eigenvector",
                              "label_propagation"]
        rows = {}
        for m in methods:
            try:
                lab = self.community(m, verbose=False, **kw)
            except Exception as e:
                rows[m] = dict(n_comm=np.nan, note=f"{type(e).__name__}: {e}")
                continue
            mix = self.mixing(lab)
            sizes = np.bincount(lab)
            rows[m] = dict(
                n_comm=len(sizes), largest=int(sizes.max()),
                singletons=int((sizes == 1).sum()),
                mean_frac_minor=round(float(mix.frac_minor.mean()), 4),
                mixed=f"{int((mix.frac_minor > 0.05).sum())}/{len(mix)}",
                note="",
            )
            self._last = getattr(self, "_last", {})
            self._last[m] = lab
        return pd.DataFrame(rows).T

    def leiden(self, resolution: float = 1.0, seed: int = 0, **kw) -> np.ndarray:
        """Back-compat wrapper; now routed through igraph."""
        return self.community("leiden", resolution=resolution, seed=seed, **kw)

    def mixing(self, labels: np.ndarray) -> pd.DataFrame:
        """Per-cluster species composition."""
        a = self.meta.get("name_a", np.unique(self.species)[0])
        df = pd.DataFrame({"cluster": labels, "species": self.species})
        out = df.pivot_table(index="cluster", columns="species",
                             aggfunc="size", fill_value=0)
        out["n"] = out.sum(axis=1)
        out["frac_minor"] = out.drop(columns="n").min(axis=1) / out["n"]
        return out.sort_values("n", ascending=False)


def dual_knn_graph(
    C: np.ndarray,
    split,
    k_intra: int | None = 15,
    k_inter: int | None = 5,
    *,
    cutoff_intra: float | None = None,
    cutoff_inter: float | None = None,
    min_intra: float | None = None,   # deprecated aliases
    min_inter: float | None = None,
    max_edges: int = 50_000_000,
    mutual_inter: bool = False,
    mutual_intra: bool = False,
    genes: np.ndarray | None = None,
    species: np.ndarray | None = None,
    block: int = 2048,
    verbose: bool = True,
) -> DualGraph:
    """Top-k_intra same-dataset and top-k_inter other-dataset edges per gene.

    split
        Either an int (first `split` rows are dataset A) or a boolean/label array
        of length n marking dataset membership.

    k_intra / k_inter
        Per-gene edge budgets. Pass None to drop the budget and keep every edge
        clearing the corresponding cutoff (a pure threshold graph). At least one
        of k or cutoff must be set for each edge class.

    cutoff_intra / cutoff_inter
        Correlation cutoffs, applied per edge class. With a k budget they act as
        a floor: a gene takes its top-k partners but only those clearing the
        cutoff, so a gene with no good partner gets no edge. Without them every
        gene is *forced* to accept k_inter cross-dataset partners no matter how
        weak, which manufactures integration — the graph looks mixed because the
        construction guaranteed it, not because the genes resemble each other.
        Pick values from the cross-dataset tail in profile_correlation's stats
        (frac_gt_0_3 / frac_gt_0_5) rather than by eye.

        `min_intra` / `min_inter` are accepted as deprecated aliases.

    mutual_inter / mutual_intra
        Keep an edge only when both endpoints proposed it. Recommended for the
        cross set — it removes hub genes that appear in everyone's top-k while
        nominating someone else.
    """
    C = np.asarray(C)
    n = C.shape[0]
    if C.ndim != 2 or C.shape[1] != n:
        raise ValueError(f"C must be square, got {C.shape}")

    if min_intra is not None:
        cutoff_intra = min_intra if cutoff_intra is None else cutoff_intra
    if min_inter is not None:
        cutoff_inter = min_inter if cutoff_inter is None else cutoff_inter
    for tag, k, cut in (("intra", k_intra, cutoff_intra), ("inter", k_inter, cutoff_inter)):
        if k is None and cut is None:
            raise ValueError(f"set k_{tag} or cutoff_{tag} (both None keeps everything)")

    if np.isscalar(split):
        is_a = np.arange(n) < int(split)
    else:
        split = np.asarray(split)
        is_a = split.astype(bool) if split.dtype == bool else split == np.unique(split)[0]
    if species is None:
        species = np.where(is_a, "A", "B")
    species = np.asarray(species)

    rows_i, cols_i, w_i = [], [], []     # intra proposals
    rows_x, cols_x, w_x = [], [], []     # inter proposals

    saturated = {"intra": 0, "inter": 0}     # genes whose k budget was binding

    def _topk(blk, mask_cols, k, floor, tag=None):
        """Top-k columns per row within mask_cols, and/or everything over floor."""
        if not mask_cols.any() or (k is not None and k <= 0):
            return None
        cand = np.flatnonzero(mask_cols)
        sub = blk[:, cand]

        if k is None:                         # pure threshold: no per-gene budget
            rr, cj = np.where(sub >= floor)
            ww = sub[rr, cj]
            if len(rr) > max_edges:
                raise MemoryError(
                    f"{len(rr):,} {tag} edges above {floor} in one block; "
                    f"raise the cutoff or set k_{tag}")
            return rr, cand[cj], ww

        kk = min(k, sub.shape[1])
        part = np.argpartition(-sub, kk - 1, axis=1)[:, :kk]
        wv = np.take_along_axis(sub, part, 1)
        order = np.argsort(-wv, axis=1)
        part = np.take_along_axis(part, order, 1)
        wv = np.take_along_axis(wv, order, 1)
        rr = np.repeat(np.arange(sub.shape[0]), kk)
        cc = cand[part.ravel()]
        ww = wv.ravel()
        keep = np.isfinite(ww)
        if floor is not None:
            keep &= ww >= floor
            if tag is not None:               # k was binding for this gene
                per_gene = keep.reshape(sub.shape[0], kk).sum(1)
                saturated[tag] += int((per_gene == kk).sum())
        elif tag is not None:
            saturated[tag] += sub.shape[0]
        return rr[keep], cc[keep], ww[keep]

    for s in range(0, n, block):
        e = min(s + block, n)
        blk = np.array(C[s:e], dtype=np.float64)
        blk[np.arange(e - s), np.arange(s, e)] = -np.inf      # no self edges
        rows_here = np.arange(s, e)

        for side in (True, False):
            rmask = is_a[s:e] == side
            if not rmask.any():
                continue
            sub = blk[rmask]
            same_cols, other_cols = (is_a, ~is_a) if side else (~is_a, is_a)

            got = _topk(sub, same_cols, k_intra, cutoff_intra, "intra")
            if got is not None:
                rr, cc, ww = got
                rows_i.append(rows_here[rmask][rr]); cols_i.append(cc); w_i.append(ww)

            got = _topk(sub, other_cols, k_inter, cutoff_inter, "inter")
            if got is not None:
                rr, cc, ww = got
                rows_x.append(rows_here[rmask][rr]); cols_x.append(cc); w_x.append(ww)

    def _build(rows, cols, ws, mutual):
        if not rows:
            return sparse.csr_matrix((n, n))
        r = np.concatenate(rows); c = np.concatenate(cols); w = np.concatenate(ws)
        P = sparse.csr_matrix((w, (r, c)), shape=(n, n))     # directed proposals
        P.sum_duplicates()
        if mutual:
            both = P.multiply(P.T != 0)
            A = both.maximum(both.T)
        else:
            A = P.maximum(P.T)
        return A.tocsr()

    A_intra = _build(rows_i, cols_i, w_i, mutual_intra)
    A_inter = _build(rows_x, cols_x, w_x, mutual_inter)
    adj = (A_intra.maximum(A_inter)).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()

    G = DualGraph(adj=adj, species=species, genes=genes,
                  meta=dict(k_intra=k_intra, k_inter=k_inter,
                            cutoff_intra=cutoff_intra, cutoff_inter=cutoff_inter,
                            saturated=dict(saturated),
                            mutual_intra=mutual_intra, mutual_inter=mutual_inter,
                            n_a=int(is_a.sum()), n_b=int((~is_a).sum()),
                            name_a=str(species[is_a][0]) if is_a.any() else "A",
                            name_b=str(species[~is_a][0]) if (~is_a).any() else "B"))
    if verbose:
        G.summary()
    return G
