"""One-to-one ortholog alignment for paired gene-correlation matrices."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd



"""
ortholog_align.py — align two gene x gene correlation matrices on 1-to-1 orthologs.

Given two square gene correlation matrices (one per species), their gene-name
lists, and an MGI/Alliance-style vertebrate homology table, this builds the set
of 1-to-1 orthologs actually present in both matrices and returns symmetrically
permuted matrices whose first X rows/cols are those orthologs in identical order.
Remaining genes are appended in their original order.

Typical notebook use
--------------------
    import numpy as np
    from ortholog_align import load_homology, align_matrices, save_aligned

    hom = load_homology("HOM_AllOrganism.rpt")

    res = align_matrices(
        mat_a=corr_mouse, genes_a=genes_mouse, species_a="mouse, laboratory",
        mat_b=corr_human, genes_b=genes_human, species_b="human",
        hom=hom,
    )
    res.summary()
    res.pairs.head()
    res.rejects.query("reason == 'multi_paralog_in_matrix'")

    save_aligned(res, "out/", name_a="mouse", name_b="human")

Design notes
------------
* 1-to-1 is evaluated *after* intersecting with the matrix gene lists, so a class
  with two paralogs in the table but only one in the matrix is usable. The strict
  table-only count is reported alongside for comparison.
* A symbol landing in more than one retained homology class is dropped — that is
  not 1-to-1 regardless of per-class counts.
* Duplicate gene symbols within a matrix gene list are dropped, never take-first.
"""

CLASS_COL = "DB Class Key"
ORG_COL = "Common Organism Name"
TAXON_COL = "NCBI Taxon ID"
SYMBOL_COL = "Symbol"
ENTREZ_COL = "EntrezGene ID"


def _norm(s: str) -> str:
    """Default symbol normalizer for cross-referencing table <-> matrix."""
    return str(s).strip().upper()


# ---------------------------------------------------------------- homology table


def load_homology(path: str, sep: str = "\t") -> pd.DataFrame:
    """Read the homology report, keeping everything as stripped strings."""
    hom = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    missing = {CLASS_COL, ORG_COL, TAXON_COL, SYMBOL_COL} - set(hom.columns)
    if missing:
        raise ValueError(f"homology table missing columns: {sorted(missing)}")
    for col in hom.columns:
        hom[col] = hom[col].str.strip()
    return hom[hom[SYMBOL_COL] != ""].copy()


def _resolve_species(hom: pd.DataFrame, key: str | int) -> str:
    """Accept a taxon ID or a common organism name; return the taxon ID."""
    key = str(key).strip()
    if key in set(hom[TAXON_COL]):
        return key
    hits = sorted(set(hom.loc[hom[ORG_COL].str.lower() == key.lower(), TAXON_COL]))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        names = sorted(set(hom[ORG_COL]))
        raise ValueError(f"species {key!r} not found. Available: {names}")
    raise ValueError(f"species {key!r} maps to multiple taxon IDs: {hits}")


# ---------------------------------------------------------------- gene lists


def _matrix_index(
    genes: Sequence[str], normalize: Callable[[str], str]
) -> tuple[dict[str, str], list[str]]:
    """Map normalized symbol -> original name. Symbols colliding are excluded."""
    counts: dict[str, int] = {}
    first: dict[str, str] = {}
    for g in genes:
        k = normalize(g)
        counts[k] = counts.get(k, 0) + 1
        first.setdefault(k, g)
    dupes = sorted(k for k, n in counts.items() if n > 1)
    lookup = {k: v for k, v in first.items() if counts[k] == 1}
    return lookup, dupes


# ---------------------------------------------------------------- ortholog logic


def one_to_one_orthologs(
    hom: pd.DataFrame,
    species_a: str | int,
    species_b: str | int,
    genes_a: Sequence[str] | None = None,
    genes_b: Sequence[str] | None = None,
    normalize: Callable[[str], str] = _norm,
    strict_multiclass: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the 1-to-1 ortholog table, optionally restricted to matrix genes.

    Returns (pairs, rejects). If genes_a/genes_b are None, the result is the
    strict table-only 1-to-1 set.

    strict_multiclass
        If True (default), a gene is disqualified up front when its symbol
        appears under more than one DB Class Key for its own species anywhere in
        the table — regardless of what the other species contributes to those
        classes. This is evaluated before per-class counting, so removing an
        ambiguous gene can leave a sibling class cleanly 1-to-1.
        If False, only classes retained in the final pair set are cross-checked
        for repeated symbols, which lets an ambiguous gene survive whenever its
        competing classes happen to fail for some other reason.

        Ambiguity is judged on the normalized symbol, matching how genes are
        looked up in the matrix: two table symbols that normalize to the same
        string are indistinguishable downstream, so both are excluded.
    """
    tax_a, tax_b = _resolve_species(hom, species_a), _resolve_species(hom, species_b)
    if tax_a == tax_b:
        raise ValueError("species_a and species_b resolve to the same taxon")

    lookup_a, dup_a = (
        _matrix_index(genes_a, normalize) if genes_a is not None else (None, [])
    )
    lookup_b, dup_b = (
        _matrix_index(genes_b, normalize) if genes_b is not None else (None, [])
    )

    cols = [CLASS_COL, TAXON_COL, SYMBOL_COL]
    if ENTREZ_COL in hom.columns:
        cols.append(ENTREZ_COL)
    sub = hom.loc[hom[TAXON_COL].isin({tax_a, tax_b}), cols].copy()
    sub["_side"] = np.where(sub[TAXON_COL].values == tax_a, "a", "b")
    sub["_n"] = sub[SYMBOL_COL].map(normalize)

    rejects: list[tuple] = []

    def _log(frame, reason):
        """Append one reject row per (class, side) with its symbols joined."""
        if frame.empty:
            return
        g = frame.groupby([CLASS_COL, "_side"], sort=False)[SYMBOL_COL].agg(
            lambda s: ";".join(sorted(set(s)))
        )
        for (ck, side), syms in g.items():
            rejects.append((ck, side, syms, reason))

    if strict_multiclass:
        nclass = sub.groupby(["_side", "_n"], sort=False)[CLASS_COL].transform("nunique")
        amb = nclass.values > 1
        if amb.any():
            bad = sub[amb]
            g = bad.groupby(["_side", "_n"], sort=False).agg(
                cks=(CLASS_COL, lambda s: ";".join(sorted(set(s)))),
                syms=(SYMBOL_COL, lambda s: ";".join(sorted(set(s)))),
            )
            for (side, _), row in g.iterrows():
                rejects.append((row["cks"], side, row["syms"], "symbol_in_multiple_classes"))
            sub = sub[~amb]

    classes = pd.Index(sub[CLASS_COL].unique(), name=CLASS_COL)

    def _counts(frame):
        c = frame.groupby([CLASS_COL, "_side"], sort=False).size().unstack(fill_value=0)
        for s in ("a", "b"):
            if s not in c.columns:
                c[s] = 0
        return c.reindex(classes, fill_value=0)

    pre = _counts(sub)

    if lookup_a is not None or lookup_b is not None:
        drop = np.zeros(len(sub), dtype=bool)
        side_arr = sub["_side"].values
        for side, lookup in (("a", lookup_a), ("b", lookup_b)):
            if lookup is None:
                continue
            drop |= (side_arr == side) & ~sub["_n"].isin(lookup).values
        subf = sub[~drop]
    else:
        subf = sub
    post = _counts(subf)

    for side, lookup in (("a", lookup_a), ("b", lookup_b)):
        absent = classes[pre[side].values == 0]
        rejects.extend((ck, side, "", "absent_from_class") for ck in absent)

        if lookup is not None:
            gone = classes[(pre[side].values > 0) & (post[side].values == 0)]
            _log(sub[sub[CLASS_COL].isin(gone) & (sub["_side"].values == side)],
                 "none_in_matrix")

        multi = classes[post[side].values > 1]
        reason = "multi_paralog_in_matrix" if lookup is not None else "multi_paralog_in_table"
        _log(subf[subf[CLASS_COL].isin(multi) & (subf["_side"].values == side)], reason)

    good = classes[(post["a"].values == 1) & (post["b"].values == 1)]
    gsub = subf[subf[CLASS_COL].isin(good)]
    side_frames = {}
    for side in ("a", "b"):
        f = gsub[gsub["_side"].values == side].set_index(CLASS_COL)
        side_frames[side] = f.reindex(good)

    pairs = pd.DataFrame({"class_key": np.asarray(good)})
    for side, lookup in (("a", lookup_a), ("b", lookup_b)):
        f = side_frames[side]
        pairs[f"symbol_{side}"] = f[SYMBOL_COL].values
        pairs[f"entrez_{side}"] = (
            f[ENTREZ_COL].values if ENTREZ_COL in f.columns else ""
        )
        norm = f["_n"].values
        pairs[f"matrix_name_{side}"] = (
            [lookup[k] for k in norm] if lookup is not None else f[SYMBOL_COL].values
        )
    pairs = pairs[
        [
            "class_key",
            "symbol_a",
            "entrez_a",
            "matrix_name_a",
            "symbol_b",
            "entrez_b",
            "matrix_name_b",
        ]
    ]

    # Safety net for strict_multiclass=False: a symbol reaching two retained classes.
    if not strict_multiclass and not pairs.empty:
        for side in ("a", "b"):
            key = pairs[f"symbol_{side}"].map(normalize)
            bad = set(key[key.duplicated(keep=False)])
            if bad:
                for _, r in pairs[key.isin(bad)].iterrows():
                    rejects.append(
                        (r["class_key"], side, r[f"symbol_{side}"], "symbol_in_multiple_classes")
                    )
                pairs = pairs[~key.isin(bad)]

    for side, dupes in (("a", dup_a), ("b", dup_b)):
        rejects.extend(("", side, d, "duplicate_symbol_in_matrix") for d in dupes)

    rejects_df = pd.DataFrame(
        rejects, columns=["class_key", "side", "symbols", "reason"]
    ).drop_duplicates()

    pairs = pairs.sort_values("class_key", key=lambda s: pd.to_numeric(s, errors="coerce"))
    return pairs.reset_index(drop=True), rejects_df.reset_index(drop=True)


# ---------------------------------------------------------------- robust filter


def max_offdiag(
    mat: np.ndarray, use_abs: bool = False, chunk: int = 1024
) -> np.ndarray:
    """Per-gene maximum correlation to any *other* gene.

    The diagonal is excluded so self-correlation never sets the value. NaNs are
    ignored; a gene whose entire row is NaN gets -inf and can never pass a
    threshold.

    Computed in row blocks and in the matrix's own dtype: a float32 20k x 20k
    input is never upcast or fully copied, which matters because a whole-matrix
    temporary at that size is 3.2 GB per copy.
    """
    mat = np.asarray(mat)
    n = mat.shape[0]
    if mat.ndim != 2 or mat.shape[1] != n:
        raise ValueError(f"matrix must be square, got {mat.shape}")
    dtype = mat.dtype if np.issubdtype(mat.dtype, np.floating) else np.float64

    out = np.empty(n, dtype=dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = mat[s:e].astype(dtype, copy=True)
        if use_abs:
            np.abs(blk, out=blk)
        blk[np.arange(e - s), np.arange(s, e)] = -np.inf
        nan = np.isnan(blk)
        if nan.any():
            np.copyto(blk, -np.inf, where=nan)
        out[s:e] = blk.max(axis=1)
    return out


def robust_subset(
    mat: np.ndarray,
    genes: Sequence[str],
    thresh: float | None,
    use_abs: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Keep only genes whose max off-diagonal correlation exceeds `thresh`.

    Returns (submatrix, subgenes, kept_index, max_corr_of_kept). `thresh=None`
    is a no-op pass-through.

    Note the set is self-consistent for a symmetric matrix: if r(X,Y) clears the
    threshold then both X and Y clear it, so every retained gene keeps the
    partner that qualified it.
    """
    mat = np.asarray(mat)
    genes = list(genes)
    mx = max_offdiag(mat, use_abs=use_abs)
    if thresh is None:
        return mat, genes, np.arange(len(genes)), mx
    keep = np.flatnonzero(mx > thresh)
    return mat[np.ix_(keep, keep)], [genes[i] for i in keep], keep, mx[keep]


# ---------------------------------------------------------------- reordering


def order_index(
    genes: Sequence[str], lead_genes: Sequence[str], keep_rest: bool = True
) -> np.ndarray:
    """Positions of `lead_genes` first, then the rest in original order."""
    pos = {}
    for i, g in enumerate(genes):
        pos.setdefault(g, i)
    lead_idx = [pos[g] for g in lead_genes]
    if len(set(lead_idx)) != len(lead_idx):
        raise ValueError("lead_genes contains duplicate matrix positions")
    if not keep_rest:
        return np.asarray(lead_idx, dtype=int)
    used = set(lead_idx)
    rest = [i for i in range(len(genes)) if i not in used]
    return np.asarray(lead_idx + rest, dtype=int)


def reorder_matrix(
    mat: np.ndarray,
    genes: Sequence[str],
    lead_genes: Sequence[str],
    keep_rest: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Symmetrically permute `mat` so `lead_genes` occupy the first rows/cols."""
    mat = np.asarray(mat)
    genes = list(genes)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"matrix must be square, got {mat.shape}")
    if mat.shape[0] != len(genes):
        raise ValueError(f"matrix is {mat.shape[0]}x{mat.shape[0]} but got {len(genes)} genes")

    idx = order_index(genes, lead_genes, keep_rest)
    return mat[np.ix_(idx, idx)], [genes[i] for i in idx], idx


# ---------------------------------------------------------------- pipeline


@dataclass
class AlignedPair:
    matrix_a: np.ndarray
    genes_a: list[str]
    matrix_b: np.ndarray
    genes_b: list[str]
    pairs: pd.DataFrame
    rejects: pd.DataFrame
    index_a: np.ndarray
    index_b: np.ndarray
    n_strict: int = 0
    max_corr_a: np.ndarray | None = None
    max_corr_b: np.ndarray | None = None
    n_input_a: int = 0
    n_input_b: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_orthologs(self) -> int:
        return len(self.pairs)

    def summary(self) -> None:
        n = self.n_orthologs
        ta, tb = self.meta.get("pcc_thresh_a"), self.meta.get("pcc_thresh_b")
        if ta is not None or tb is not None:
            print(f"robust filter A (max r > {ta}): "
                  f"{self.meta.get('n_robust_a')} / {self.n_input_a} genes")
            print(f"robust filter B (max r > {tb}): "
                  f"{self.meta.get('n_robust_b')} / {self.n_input_b} genes")
        if self.n_strict:
            print(f"1-to-1 orthologs (table only):        {self.n_strict}")
        print(f"1-to-1 orthologs (robust in both):    {n}")
        print(f"  classes still multi-paralog:        "
              f"{(self.rejects.reason == 'multi_paralog_in_matrix').sum()}")
        print(f"matrix A: {self.matrix_a.shape}, first {n} rows/cols are orthologs")
        print(f"matrix B: {self.matrix_b.shape}, first {n} rows/cols are orthologs")
        print("\nrejection reasons:")
        print(self.rejects.reason.value_counts().to_string())

    def gene_table(self, side: str = "a") -> pd.DataFrame:
        """Per-gene table for one dataset, in final matrix order."""
        genes = self.genes_a if side == "a" else self.genes_b
        mx = self.max_corr_a if side == "a" else self.max_corr_b
        n = self.n_orthologs
        return pd.DataFrame(
            {
                "position": np.arange(len(genes)),
                "gene": genes,
                "max_corr": mx if mx is not None else np.nan,
                "in_ortholog_block": np.arange(len(genes)) < n,
            }
        )

    def check(self, tol: float = 1e-8) -> None:
        """Assert the ortholog blocks line up and symmetry survived."""
        n = self.n_orthologs
        assert self.genes_a[:n] == list(self.pairs.matrix_name_a), "block A misordered"
        assert self.genes_b[:n] == list(self.pairs.matrix_name_b), "block B misordered"
        rng = np.random.default_rng(0)
        for m, tag in ((self.matrix_a, "A"), (self.matrix_b, "B")):
            k = m.shape[0]
            if k == 0:
                continue
            # sampled rather than full: allclose on a 20k x 20k matrix allocates
            # several more copies of it just to answer a diagnostic question
            i = rng.choice(k, size=min(k, 512), replace=False)
            blk = m[np.ix_(i, i)]
            if np.allclose(blk, blk.T, atol=tol, equal_nan=True):
                continue
            print(f"note: matrix {tag} is not symmetric (permutation still valid)")
        print(f"OK — {n} aligned ortholog positions")


def align_matrices(
    mat_a: np.ndarray,
    genes_a: Sequence[str],
    species_a: str | int,
    mat_b: np.ndarray,
    genes_b: Sequence[str],
    species_b: str | int,
    hom: pd.DataFrame,
    keep_rest: bool = True,
    normalize: Callable[[str], str] = _norm,
    strict_multiclass: bool = True,
    pcc_thresh_a: float | None = None,
    pcc_thresh_b: float | None = None,
    use_abs: bool = False,
    report_strict: bool = True,
) -> AlignedPair:
    """Full pipeline: robust filter -> orthologs -> symmetric permutation.

    pcc_thresh_a / pcc_thresh_b
        Per-dataset cutoffs on each gene's maximum off-diagonal correlation.
        Genes at or below the cutoff are dropped from that dataset entirely, so
        the returned matrices are the "robust" submatrices. Ortholog matching
        then runs on the survivors, meaning the leading block is orthologs that
        are robust in *both* datasets and the tail is each dataset's remaining
        robust genes. `None` disables filtering for that dataset.

    use_abs
        Score on |r| instead of signed r, if strong anticorrelation should also
        qualify a gene as robust.

    report_strict
        Also compute the table-only 1-to-1 count for the summary. Set False to
        skip a second pass over the homology table.
    """
    n_in_a, n_in_b = len(genes_a), len(genes_b)

    def _prep(mat, genes, thresh):
        mx = max_offdiag(mat, use_abs=use_abs)
        keep = np.arange(len(genes)) if thresh is None else np.flatnonzero(mx > thresh)
        return mx, keep, [genes[i] for i in keep]

    mx_a, keep_a, rgenes_a = _prep(mat_a, genes_a, pcc_thresh_a)
    mx_b, keep_b, rgenes_b = _prep(mat_b, genes_b, pcc_thresh_b)

    kw = dict(normalize=normalize, strict_multiclass=strict_multiclass)
    n_strict = 0
    if report_strict:
        n_strict = len(one_to_one_orthologs(hom, species_a, species_b, **kw)[0])
    pairs, rejects = one_to_one_orthologs(
        hom, species_a, species_b, rgenes_a, rgenes_b, **kw
    )

    # One fused gather per matrix: robust filter and ortholog ordering combined,
    # so a large matrix is copied once rather than twice.
    loc_a = order_index(rgenes_a, list(pairs.matrix_name_a), keep_rest)
    loc_b = order_index(rgenes_b, list(pairs.matrix_name_b), keep_rest)
    idx_a, idx_b = keep_a[loc_a], keep_b[loc_b]

    new_a = np.asarray(mat_a)[np.ix_(idx_a, idx_a)]
    new_b = np.asarray(mat_b)[np.ix_(idx_b, idx_b)]
    ga = [rgenes_a[i] for i in loc_a]
    gb = [rgenes_b[i] for i in loc_b]

    return AlignedPair(
        matrix_a=new_a,
        genes_a=ga,
        matrix_b=new_b,
        genes_b=gb,
        pairs=pairs,
        rejects=rejects,
        index_a=idx_a,
        index_b=idx_b,
        n_strict=n_strict,
        max_corr_a=mx_a[idx_a],
        max_corr_b=mx_b[idx_b],
        n_input_a=n_in_a,
        n_input_b=n_in_b,
        meta={
            "species_a": str(species_a),
            "species_b": str(species_b),
            "keep_rest": keep_rest,
            "strict_multiclass": strict_multiclass,
            "pcc_thresh_a": pcc_thresh_a,
            "pcc_thresh_b": pcc_thresh_b,
            "use_abs": use_abs,
            "n_robust_a": len(rgenes_a),
            "n_robust_b": len(rgenes_b),
        },
    )


# ---------------------------------------------------------------- io


def save_aligned(
    res: AlignedPair, outdir: str, name_a: str = "a", name_b: str = "b"
) -> dict[str, str]:
    """Write matrices (.npy), gene lists (.txt), ortholog table and rejects (.tsv)."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    for tag, mat, genes in ((name_a, res.matrix_a, res.genes_a), (name_b, res.matrix_b, res.genes_b)):
        mp = os.path.join(outdir, f"{tag}_corr_orthoaligned.npy")
        gp = os.path.join(outdir, f"{tag}_genes_orthoaligned.txt")
        np.save(mp, mat)
        with open(gp, "w") as fh:
            fh.write("\n".join(genes) + "\n")
        paths[f"{tag}_matrix"], paths[f"{tag}_genes"] = mp, gp

    for tag, side in ((name_a, "a"), (name_b, "b")):
        tp = os.path.join(outdir, f"{tag}_gene_table.tsv")
        res.gene_table(side).to_csv(tp, sep="\t", index=False)
        paths[f"{tag}_table"] = tp

    pp = os.path.join(outdir, "ortholog_pairs.tsv")
    rp = os.path.join(outdir, "ortholog_rejects.tsv")
    res.pairs.to_csv(pp, sep="\t", index=False)
    res.rejects.to_csv(rp, sep="\t", index=False)
    paths["pairs"], paths["rejects"] = pp, rp

    print(f"wrote {len(paths)} files to {outdir} "
          f"(n_orthologs={res.n_orthologs}, block = first {res.n_orthologs} rows/cols)")
    return paths
