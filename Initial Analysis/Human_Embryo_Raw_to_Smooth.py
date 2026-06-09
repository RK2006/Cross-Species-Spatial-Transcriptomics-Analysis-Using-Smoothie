import os
import sys
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import smoothie
smoothie.suppress_warnings()

filename = sys.argv[1]
input_path = os.path.join("../data", filename)
stem, ext = os.path.splitext(filename)
output_path = os.path.join("../data", stem + "_smooth" + ext)

adata = ad.read_h5ad(input_path)

sc.pp.filter_cells(adata, min_counts=1)
sc.pp.filter_genes(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=10)

sc.pp.log1p(adata)

print(adata.shape)

target_microns = 40.0
micron_to_unit_conversion = 2
sm_adata = smoothie.run_parallelized_smoothing(
    adata,
    grid_based_or_not=True,
    gaussian_sd=target_microns * micron_to_unit_conversion,
    min_spots_under_gaussian=1000
)

print(f"Writing smoothed data to {output_path}")
sm_adata.write_h5ad(output_path)

print(sm_adata.shape)