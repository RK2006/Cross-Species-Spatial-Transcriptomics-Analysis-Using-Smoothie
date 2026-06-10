import os
import sys
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import smoothie
smoothie.suppress_warnings()


def compute_and_save_correlation(sm_adata, stem):
    pearsonR_mat, p_val_mat = smoothie.compute_correlation_matrix(sm_adata.X)
    np.save(f"../data/{stem}_pearsonR.npy", pearsonR_mat)
    np.save(f"../data/{stem}_pval.npy", p_val_mat)
    np.save(f"../data/{stem}_gene_names.npy", sm_adata.var_names.to_numpy())
    return pearsonR_mat, p_val_mat


def create_spatial_network(sm_adata, pearsonR_mat, pcc_cutoff=0.4, clustering_power=3, output_folder='../Cytoscape2'):
    edge_list, node_label_df = smoothie.make_spatial_network(
        pearsonR_mat=pearsonR_mat,  # don't change
        gene_names=sm_adata.var_names,  # don't change
        pcc_cutoff=pcc_cutoff,
        clustering_power=clustering_power,
        output_folder=output_folder
    )
    return edge_list, node_label_df


def plot_modules(sm_adata, node_label_df, output_folder='./module_plots', min_genes=3, spot_size=50, plots_per_row=10, dpi=300):
    smoothie.plot_modules(
        sm_adata,
        node_label_df,
        output_folder=output_folder,
        min_genes=min_genes,
        spot_size=spot_size,
        plots_per_row=plots_per_row,
        dpi=dpi
    )


if __name__ == "__main__":
    filename = sys.argv[1]
    input_path = os.path.join("../data", filename)
    stem = os.path.splitext(filename)[0]

    sm_adata = ad.read_h5ad(input_path)
    print(sm_adata.shape)

    pearsonR_mat, p_val_mat = compute_and_save_correlation(sm_adata, stem)
    edge_list, node_label_df = create_spatial_network(sm_adata, pearsonR_mat)
    # plot_modules(sm_adata, node_label_df)
