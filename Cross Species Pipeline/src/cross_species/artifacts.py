"""Save and load self-describing core pipeline runs."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .graph import DualGraph
from .representation import EdgeSets, StitchedProfiles

SCHEMA_VERSION = 1


def _jsonable(value):
    """Convert NumPy/path values in metadata to ordinary JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def gene_order_hash(genes, species) -> str:
    """Stable digest used to reject artifacts with a different node order."""
    digest = sha256()
    for species_name, gene in zip(np.asarray(species), np.asarray(genes)):
        digest.update(str(species_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(gene).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    prof,
    *,
    graph_parameters: dict[str, Any],
    resolution: float,
    leiden_seed: int,
    input_paths: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Describe the inputs and ordering needed by downstream experiments."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_genes": int(len(prof.genes)),
        "n_orthologs": int(prof.meta["n_ortho"]),
        "species_split": int(prof.meta["width_a"]),
        "species_names": [str(prof.meta["name_a"]), str(prof.meta["name_b"])],
        "gene_order_sha256": gene_order_hash(prof.genes, prof.species),
        "profile_shape": [int(value) for value in prof.z.shape],
        "similarity": "Pearson correlation of Fisher-z ortholog-correlation profiles",
        "graph_parameters": _jsonable(dict(graph_parameters)),
        "leiden_resolution": float(resolution),
        "leiden_seed": int(leiden_seed),
        "input_paths": {} if input_paths is None else {
            key: str(value) for key, value in input_paths.items()
        },
        "notes": notes,
    }


def save_core_run(
    output_dir,
    *,
    prof,
    C,
    graph,
    labels,
    graph_parameters,
    resolution,
    leiden_seed,
    edge_sets=None,
    ortholog_pairs=None,
    input_paths=None,
    notes=None,
):
    """Write the complete representation, graph, clustering, and manifest."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Core output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels)
    C = np.asarray(C)
    n_genes = len(prof.genes)
    if C.shape != (n_genes, n_genes):
        raise ValueError("C and prof describe different gene universes")
    if graph.adj.shape != C.shape or len(labels) != n_genes:
        raise ValueError("graph, labels, C, and prof must use the same gene order")
    if int(prof.meta["n_ortho"]) != int(prof.z.shape[0]):
        raise ValueError("prof metadata disagrees with its ortholog-profile matrix")

    np.save(output_dir / "profiles_z.npy", np.asarray(prof.z))
    np.save(output_dir / "genes.npy", np.asarray(prof.genes))
    np.save(output_dir / "raw_genes.npy", np.asarray(prof.raw_genes))
    np.save(output_dir / "species.npy", np.asarray(prof.species))
    np.save(output_dir / "similarity.npy", C)
    sparse.save_npz(output_dir / "graph_adjacency.npz", graph.adj.tocsr())
    np.save(output_dir / "labels.npy", labels)

    manifest = build_manifest(
        prof,
        graph_parameters=graph_parameters,
        resolution=resolution,
        leiden_seed=leiden_seed,
        input_paths=input_paths,
        notes=notes,
    )
    manifest["graph_metadata"] = _jsonable(graph.meta)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame({
        "node_index": np.arange(n_genes),
        "gene": prof.genes,
        "raw_gene": prof.raw_genes,
        "species": prof.species,
        "module": labels,
    }).to_csv(output_dir / "nodes.csv", index=False)
    graph.to_csv(
        output_dir / "edges.csv.gz",
        labels=labels,
        index_cols=True,
        verbose=False,
    )
    if edge_sets is not None:
        edge_sets.stats.to_csv(
            output_dir / "similarity_distribution_summary.csv", index=False
        )
        np.savez_compressed(
            output_dir / "similarity_distributions.npz",
            bins=np.asarray(edge_sets.bins),
            **{f"counts_{key}": np.asarray(value) for key, value in edge_sets.counts.items()},
        )
        (output_dir / "similarity_metadata.json").write_text(
            json.dumps(edge_sets.meta, indent=2)
        )
    if ortholog_pairs is not None:
        ortholog_pairs.to_csv(output_dir / "ortholog_pairs.csv", index=False)
    print(f"Core run written to: {output_dir}")
    return manifest


def load_core_run(output_dir, *, mmap_mode="r"):
    """Load a core run and verify its shape, split, and gene-order digest."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing core manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported core schema {manifest.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )

    genes = np.load(output_dir / "genes.npy", allow_pickle=False)
    raw_genes = np.load(output_dir / "raw_genes.npy", allow_pickle=False)
    species = np.load(output_dir / "species.npy", allow_pickle=False)
    z = np.load(output_dir / "profiles_z.npy", mmap_mode=mmap_mode)
    C = np.load(output_dir / "similarity.npy", mmap_mode=mmap_mode)
    labels = np.load(output_dir / "labels.npy", allow_pickle=False)
    adjacency = sparse.load_npz(output_dir / "graph_adjacency.npz").tocsr()

    expected_n = int(manifest["n_genes"])
    expected_orthologs = int(manifest["n_orthologs"])
    if not (
        len(genes) == len(raw_genes) == len(species) == len(labels) == expected_n
        and z.shape == (expected_orthologs, expected_n)
        and C.shape == (expected_n, expected_n)
        and adjacency.shape == (expected_n, expected_n)
    ):
        raise ValueError("Saved core artifacts do not match manifest dimensions")
    observed_hash = gene_order_hash(genes, species)
    if observed_hash != manifest["gene_order_sha256"]:
        raise ValueError("Saved gene ordering does not match the core manifest")

    name_a, name_b = manifest["species_names"]
    split = int(manifest["species_split"])
    if not (0 < split < expected_n):
        raise ValueError("Saved species split is outside the gene array")
    if not (
        np.all(species[:split].astype(str) == str(name_a))
        and np.all(species[split:].astype(str) == str(name_b))
    ):
        raise ValueError("Saved species ordering does not match the manifest split")
    prof = StitchedProfiles(
        z=z,
        genes=genes,
        species=species,
        raw_genes=raw_genes,
        meta={
            "n_ortho": expected_orthologs,
            "width_a": split,
            "name_a": name_a,
            "name_b": name_b,
            "loaded_from": str(output_dir),
        },
    )
    edge_sets = None
    distributions_path = output_dir / "similarity_distributions.npz"
    stats_path = output_dir / "similarity_distribution_summary.csv"
    metadata_path = output_dir / "similarity_metadata.json"
    if distributions_path.exists() and stats_path.exists() and metadata_path.exists():
        archive = np.load(distributions_path)
        counts = {
            key.removeprefix("counts_"): archive[key]
            for key in archive.files if key.startswith("counts_")
        }
        edge_sets = EdgeSets(
            bins=archive["bins"],
            counts=counts,
            stats=pd.read_csv(stats_path),
            meta=json.loads(metadata_path.read_text()),
        )
    ortholog_pairs_path = output_dir / "ortholog_pairs.csv"
    ortholog_pairs = (
        pd.read_csv(ortholog_pairs_path, dtype={"class_key": str})
        if ortholog_pairs_path.exists() else None
    )
    if ortholog_pairs is not None and len(ortholog_pairs) != expected_orthologs:
        raise ValueError("Saved ortholog pairs do not match the profile feature count")

    graph = DualGraph(
        adjacency,
        species,
        genes,
        meta={
            **manifest.get("graph_metadata", manifest["graph_parameters"]),
            "name_a": name_a,
            "name_b": name_b,
            "loaded_from": str(output_dir),
        },
    )
    return {
        "manifest": manifest,
        "prof": prof,
        "C": C,
        "graph": graph,
        "labels": labels,
        "edge_sets": edge_sets,
        "ortholog_pairs": ortholog_pairs,
        "output_dir": str(output_dir),
    }
