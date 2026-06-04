import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import smoothie
smoothie.suppress_warnings()

sm_adata = ad.read_h5ad("../data/GSM9629357_CS23_E2S1_smooth_bin1.h5ad")

print(sm_adata.shape)

pearsonR_mat, p_val_mat = smoothie.compute_correlation_matrix(sm_adata.X)

np.save("../data/GSM9629357_CS23_E2S1_bin1_pearsonR.npy", pearsonR_mat)
np.save("../data/GSM9629357_CS23_E2S1_bin1_pval.npy", p_val_mat)
np.save("../data/GSM9629357_CS23_E2S1_bin_1gene_names.npy", sm_adata.var_names.to_numpy())

edge_list, node_label_df = smoothie.make_spatial_network(
    pearsonR_mat=pearsonR_mat, # don't change
    gene_names=sm_adata.var_names, # don't change
    pcc_cutoff=0.4,
    clustering_power=3,
    output_folder='../Cytoscape'
)