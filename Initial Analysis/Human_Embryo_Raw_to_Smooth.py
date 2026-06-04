import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import smoothie
smoothie.suppress_warnings()

sm_adata = adata = ad.read_h5ad("../data/GSM9629357_CS23_E2S1_raw_bin1.h5ad")

sc.pp.filter_cells(adata, min_counts=1)
sc.pp.filter_genes(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=10)

sc.pp.log1p(adata)

print(adata.shape)

target_microns = 20.0
micron_to_unit_conversion = 2
sm_adata = smoothie.run_parallelized_smoothing(
    adata,
    grid_based_or_not=True,
    gaussian_sd=target_microns * micron_to_unit_conversion,
    min_spots_under_gaussian=25
)

sm_adata.write_h5ad("../data/GSM9629357_CS23_E2S1_smooth_bin1.h5ad")