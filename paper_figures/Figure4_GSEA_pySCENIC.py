#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import gseapy as gp

# Load DEG table
deg = pd.read_csv("./Module_specific_gene_expression.csv")

# Standardize column names
deg = deg.rename(columns={"avg_log2FC": "log2FC"})

# Get GO gene sets
msig = gp.Msigdb()
gmt = msig.get_gmt(category='m5.go', dbver="2024.1.Mm")

# Get all module names
module_names = sorted(deg['cluster'].unique())

# Main loop
for module in module_names:
    # Filter genes for current module
    sub_deg = deg[deg['cluster'] == module].copy()
    # Sort by log2FC
    ranking = sub_deg[['gene', 'log2FC']].sort_values('log2FC', ascending=False).reset_index(drop=True)
    
    # Create output directory
    outdir = f'./GSEA_{module}'
    os.makedirs(outdir, exist_ok=True)
    
    # Run pre-ranked GSEA
    pre_res = gp.prerank(
        rnk=ranking,
        gene_sets=gmt,
        seed=6,
        permutation_num=1000,
        min_size=1,
        max_size=10000,
        outdir=outdir,
        format='png'
    )
    
    # Extract results
    out = []
    for term in pre_res.results:
        out.append([
            term,
            pre_res.results[term]['fdr'],
            pre_res.results[term]['es'],
            pre_res.results[term]['nes']
        ])
    out_df = pd.DataFrame(out, columns=['Term', 'fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop=True)
    # Save results
    out_df.to_csv(os.path.join(outdir, f"{module}_GSEA_summary.csv"), index=False)
    print(f"{module}: Done.")

print('All modules finished!')




# pySCENIC

# Run in R
# exprMat <- as.matrix(GetAssayData(scRNA_harmony,slot = "counts"))

# output matrix
# write.csv(t(exprMat),file="./scenic.data.csv")


# Create loom file
# Write the data saved in "scenic.data.csv" in R to the loom file
import scanpy as sc
import numpy as np
import loompy as lp

# Read csv data
x = sc.read_csv("scenic.data.csv")

# Process data
row_attrs = {"Gene": np.array(x.var_names)}
col_attrs = {"CellID": np.array(x.obs_names)}

# Write loom file
lp.create("scenic.loom", x.X.transpose(), row_attrs, col_attrs)


# Run in bash

#!/bin/bash

# # Step 1: grn
# echo "Start grn step: $(date)"
# pyscenic grn --num_workers 10 --output grn.tsv --method grnboost2 scenic.loom mm_mgi_tfs.txt
# echo "Finish grn step: $(date)"

# # Step 2: ctx
# echo "Start ctx step: $(date)"
# pyscenic ctx \
#   grn.tsv mm9-tss-centered-5kb-7species.mc9nr.genes_vs_motifs.rankings.feather \
#   --annotations_fname motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl \
#   --expression_mtx_fname scenic.loom \
#   --mode "dask_multiprocessing" \
#   --output ctx.csv \
#   --num_workers 10 \
#   --mask_dropouts
# echo "Finish ctx step: $(date)"

# # Step 3: aucell
# echo "Start aucell step: $(date)"
# pyscenic aucell \
#   scenic.loom \
#   ctx.csv \
#   --output aucell.loom \
#   --num_workers 10
# echo "Finish aucell step: $(date)"

# echo "All steps finished: $(date)"





