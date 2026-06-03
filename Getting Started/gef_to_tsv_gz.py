import h5py
import numpy as np
import pandas as pd

# Path to gef data
gef_path = "data/GSM9629357_CS23_E2S1.gef"
output_path = "data/GSM9629357_CS23_E2S1_bin1.tsv.gz"

print("Opening GEF file...")
f = h5py.File(gef_path, "r")

print("Loading expression and gene data...")
expr  = f["geneExp/bin1/expression"][:]
genes = f["geneExp/bin1/gene"][:]

f.close()

print("Decoding gene names...")
gene_names = np.array([
    g.decode("utf-8") if isinstance(g, bytes) else g
    for g in genes["gene"]
])
offsets = genes["offset"].astype(np.int64)
counts  = genes["count"].astype(np.int64)

x_all   = expr["x"].astype(np.int32)
y_all   = expr["y"].astype(np.int32)
mid_all = expr["count"].astype(np.int32)

print(f"Genes: {len(gene_names)}, total expression records: {len(expr)}")

# Expand gene name for each expression record using offset/count index
print("Expanding gene labels...")
gene_col = np.empty(len(expr), dtype=object)
for i, (off, cnt) in enumerate(zip(offsets, counts)):
    gene_col[off : off + cnt] = gene_names[i]

print(f"Writing {output_path}...")
df = pd.DataFrame({
    "x":         x_all,
    "y":         y_all,
    "geneID":    gene_col,
    "MIDCounts": mid_all,
})
df.to_csv(output_path, sep="\t", index=False, compression="gzip")
print(f"Saved to {output_path}")
