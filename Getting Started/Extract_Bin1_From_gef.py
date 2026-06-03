import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

# Path to gef data
gef_path = "data/GSM9629357_CS23_E2S1.gef"
output_path = "data/GSM9629357_CS23_E2S1.h5ad"

print("Opening GEF file...")
f = h5py.File(gef_path, "r")

print("Loading expression and gene data...")
expr = f["geneExp/bin1/expression"][:]
genes = f["geneExp/bin1/gene"][:]

# expr fields: x, y, MIDCount (UMI count per spot per gene)
# genes fields: geneName, offset, count (range into expr for each gene)

# Decode gene names from bytes if needed
print("Decoding gene names...")
gene_names_raw = [
    g.decode("utf-8") if isinstance(g, bytes) else g
    for g in genes["gene"]
]
offsets = genes["offset"].astype(np.int64)
counts = genes["count"].astype(np.int64)

print("Sorting genes alphabetically...")
sort_order = np.argsort(gene_names_raw)
gene_names = [gene_names_raw[i] for i in sort_order]
offsets = offsets[sort_order]
counts = counts[sort_order]

x_coords = expr["x"].astype(np.int64)
y_coords = expr["y"].astype(np.int64)
mid_counts = expr["count"].astype(np.float32)

# Build unique spot index from (x, y) pairs
print("Building unique spot index from (x, y) coordinates...")
spots = np.stack([x_coords, y_coords], axis=1)
unique_spots, spot_indices = np.unique(spots, axis=0, return_inverse=True)
n_spots = len(unique_spots)
n_genes = len(gene_names)

print(f"Spots: {n_spots}, Genes: {n_genes}, Total expr entries: {len(expr)}")

# Build COO sparse matrix: rows=spots, cols=genes
print("Building sparse count matrix (this may take a while)...")
row_indices = []
col_indices = []
data_values = []

for gene_idx in range(n_genes):
    start = offsets[gene_idx]
    end = start + counts[gene_idx]
    gene_spot_indices = spot_indices[start:end]
    gene_mid_counts = mid_counts[start:end]
    row_indices.append(gene_spot_indices)
    col_indices.append(np.full(len(gene_spot_indices), gene_idx, dtype=np.int32))
    data_values.append(gene_mid_counts)

print("Concatenating indices and converting to CSR format...")
row_indices = np.concatenate(row_indices)
col_indices = np.concatenate(col_indices)
data_values = np.concatenate(data_values)

X = sp.csr_matrix(
    (data_values, (row_indices, col_indices)),
    shape=(n_spots, n_genes),
    dtype=np.float32,
)

# Build AnnData
print("Building AnnData object...")
adata = ad.AnnData(X=X)
adata.var_names = gene_names
adata.obsm["spatial"] = unique_spots.astype(np.float32)  # float32 matches Smoothie

f.close()

print(f"Writing to {output_path}...")
adata.write_h5ad(output_path)
print(f"Saved to {output_path}")
print(adata)
