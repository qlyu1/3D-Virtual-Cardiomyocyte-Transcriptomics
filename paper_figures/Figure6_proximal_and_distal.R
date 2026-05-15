## Load required libraries
library(dplyr)
library(Seurat)
library(ggplot2)
library(RColorBrewer)
library(patchwork)
library(viridis)
library(DoubletFinder)
library(data.table)
library(ggrepel)
library(hdf5r)
library(tidyverse)
library(Matrix)
library(ggrastr)
library(ggnewscale)
library(grid)

## Color palette
mycol <- c(
  "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#8CD17D",
  "#B6992D", "#499894", "#E15759", "#FF9D9A", "#79706E",
  "#D37295", "#FABFD2", "#B07AA1", "#D4A6C8", "#9D7660",
  "#E58606", "#5D69B1", "#24796C", "#DAA51B", "#000000",
  "#99C945", "#ED645A",
  "#66C5CC", "#F6CF71", "#F89C74", "#DCB0F2", "#87C55F",
  "#9EB9F3", "#FE88B1", "#C9DB74", "#8BE0A4", "#B497E7",
  "#D3B484", "#B4D2BA", "#DE6E6E", "#5A9E88", "#FF6F91",
  "#7C90F2", "#C3A38A","#FFD121","#088247")

setwd('/Volumes/T9/v4/expression-0121/05_polarity_10layers')

## ============================================================================
## Helper function: Seurat normalization and dimensionality reduction
## ============================================================================

seurat_standard_normalize_and_scale <- function(colon, cluster = FALSE, cluster_resolution = 0.5) {
  colon <- NormalizeData(colon, normalization.method = "LogNormalize", scale.factor = 10000)
  colon <- FindVariableFeatures(colon, selection.method = "vst", nfeatures = 2000)
  all.genes <- rownames(colon)
  colon <- ScaleData(colon, features = all.genes)
  colon <- RunPCA(colon, features = VariableFeatures(object = colon), npcs = 20)
  if (cluster) {
    colon <- FindNeighbors(colon, dims = 1:20)
    colon <- FindClusters(colon, resolution = cluster_resolution)
  }
  colon <- RunUMAP(colon, dims = 1:20)
  return(colon)
}

## ============================================================================
## Part 1: Data loading and initial processing
## ============================================================================

data_directory <- c('module1_cells_contour_expression_symbols.csv')
colon.data <- read.csv(file = data_directory, header = TRUE, row.names = 1)
project_name <- 'Tracks'

## Create Seurat object
currentSample <- CreateSeuratObject(counts = colon.data, project = project_name, 
                                   min.cells = 0, min.features = 0)
currentSample[["percent.mt"]] <- PercentageFeatureSet(currentSample, pattern = "^mt-")

## QC plots
pdf(paste0("./result/qc_plots_", project_name, "_prefiltered.pdf"))
print(VlnPlot(currentSample, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
              ncol = 3, pt.size = 0.05))
dev.off()

pdf(paste0("./result/qc_plots_", project_name, "_prefiltered_no_points.pdf"))
print(VlnPlot(currentSample, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
              ncol = 3, pt.size = 0))
dev.off()

## Normalize and process
currentSample <- seurat_standard_normalize_and_scale(currentSample, FALSE)
seu_colon <- DietSeurat(currentSample, counts = TRUE, data = TRUE, 
                        scale.data = FALSE, assays = "RNA")

scRNA_harmony <- seu_colon

## Extract slice information from cell barcodes
bc <- rownames(scRNA_harmony@meta.data)
scRNA_harmony$Slices <- as.integer(sub(".*\\.T\\d+S(\\d+)\\..*", "\\1", bc))
scRNA_harmony$Slices <- factor(scRNA_harmony$Slices)

## ============================================================================
## Part 2: 10-layer analysis
## ============================================================================

sc.subtype.1 <- scRNA_harmony

## Filter cells by UMI count quantiles
meta <- sc.subtype.1@meta.data
lower_bound <- quantile(meta$nCount_RNA, 0.15)
upper_bound <- quantile(meta$nCount_RNA, 0.75)
cells_to_keep <- rownames(meta)[meta$nCount_RNA >= lower_bound & meta$nCount_RNA <= upper_bound]
sc.subtype.1 <- subset(sc.subtype.1, cells = cells_to_keep)

## Standard processing pipeline
sc.subtype.1 <- NormalizeData(sc.subtype.1) %>% 
  FindVariableFeatures() %>% 
  ScaleData(vars.to.regress = c("nCount_RNA")) %>% 
  RunPCA(verbose = FALSE)

sc.subtype.1 <- FindNeighbors(sc.subtype.1, dims = 1:10) %>% 
  FindClusters(resolution = 0.6)
sc.subtype.1 <- RunUMAP(sc.subtype.1, dims = 1:10)

Idents(sc.subtype.1) <- 'Slices'

VlnPlot(sc.subtype.1, features = c("nCount_RNA", "nFeature_RNA", 'percent.mt'), 
        group.by = "Slices")

save(sc.subtype.1, file = 'sc.subtype_10_layers.Rda')

meta <- sc.subtype.1@meta.data
write.csv(meta, file = 'sc_subtype_10_layers_metadata.csv')

## ============================================================================
## Part 3: Linear regression analysis for polarity genes
## ============================================================================

meta <- sc.subtype.1@meta.data
meta$Slices <- as.numeric(meta$Slices)
stopifnot(!anyNA(meta$Slices))

## Extract log-normalized expression
expr <- GetAssayData(sc.subtype.1, assay = "RNA", layer = "data")
expr <- expr[, rownames(meta), drop = FALSE]

## Filter low-expressed genes
keep <- Matrix::rowSums(expr > 0) >= 10
expr <- expr[keep, , drop = FALSE]

## Linear regression: expression ~ Slices + covariates
res <- t(apply(as.matrix(expr), 1, function(y) {
  df <- meta
  df$y <- as.numeric(y)
  m <- lm(y ~ Slices + log10(nCount_RNA + 1) + percent.mt, data = df)
  cc <- summary(m)$coefficients
  c(slope = cc["Slices", "Estimate"], pval = cc["Slices", "Pr(>|t|)"])
}))

results_df <- data.frame(
  gene  = rownames(res),
  slope = res[, "slope"],
  pval  = res[, "pval"],
  padj  = p.adjust(res[, "pval"], method = "BH"),
  row.names = NULL
)

write.csv(results_df, file = '10layers_lm_polarity_genes_mt.csv')

## ============================================================================
## Part 4: Volcano plot for polarity genes
## ============================================================================

genes <- c("Ryr2","Prkca","Prkch","Myh7","Nppb","Cd36","Acsl1","Acadm","Hrc","Gja1",
           "Tnnt2","Tnni3","Myh6","Myl2","Lrrc10","Tcap","Ckm","Perm1","Tmem38a","Obscn")

df <- results_df %>%
  mutate(
    neglog10_padj = -log10(padj + 1e-300),
    sig = padj < 0.05
  )

top_n <- 0
lab_genes_top <- bind_rows(
  df %>% filter(padj < 0.05) %>% arrange(desc(slope)) %>% slice_head(n = top_n),
  df %>% filter(padj < 0.05) %>% arrange(slope) %>% slice_head(n = top_n)
) %>% distinct(gene, .keep_all = TRUE)

lab_genes_forced <- df %>%
  filter(gene %in% genes) %>%
  distinct(gene, .keep_all = TRUE)

lab_genes <- bind_rows(lab_genes_top, lab_genes_forced) %>%
  distinct(gene, .keep_all = TRUE)

df <- df %>%
  mutate(slope_dir = case_when(
    padj < 0.05 & slope > 0 ~ "pos",
    padj < 0.05 & slope < 0 ~ "neg",
    TRUE ~ "ns"
  ))

p_volcano <- ggplot(df, aes(slope, neglog10_padj)) +
  rasterise(
    geom_point(aes(color = slope_dir), alpha = 0.7, size = 1.6),
    dpi = 300
  ) +
  geom_point(
    data = lab_genes,
    aes(slope, neglog10_padj),
    inherit.aes = FALSE,
    shape = 21,
    fill = NA,
    color = "black",
    stroke = 0.25,
    size = 1.9
  ) +
  geom_vline(xintercept = 0, linetype = 2, color = "grey50") +
  geom_hline(yintercept = -log10(0.05), linetype = 2, color = "grey50") +
  geom_text_repel(
    data = lab_genes,
    aes(label = gene),
    size = 2,
    max.overlaps = Inf,
    box.padding = 0.3,
    segment.size = 0.2,
    segment.color = "grey30"
  ) +
  scale_color_manual(
    values = c(pos = "#D62728", neg = "#1F77B4", ns = "grey70"),
    breaks = c("pos", "neg", "ns"),
    labels = c("FDR<0.05, slope>0", "FDR<0.05, slope<0", "NS")
  ) +
  coord_cartesian(ylim = c(0, 160)) +
  labs(x = "Slope (Slices effect)", y = "-log10(FDR)", color = NULL) +
  theme_classic()

ggsave("volcano.pdf", p_volcano, width = 5, height = 3.5, device = cairo_pdf)

## ============================================================================
## Part 5: Differential expression analysis (top vs bottom layers)
## ============================================================================

load('sc.subtype_10_layers.Rda')

## Define position groups
sc.subtype.1$Position <- ifelse(
  sc.subtype.1$Slices %in% c("1", "2", "3"), "bottom",
  ifelse(sc.subtype.1$Slices %in% c("8", "9", "10"), "top", "others")
)

Idents(sc.subtype.1) <- 'Position'

## Find markers for all groups
markers <- FindAllMarkers(sc.subtype.1, only.pos = TRUE, min.pct = 0.1, 
                         logfc.threshold = 0.1)
markers <- markers[markers$p_val_adj < 0.05, ]
write.csv(markers, file = 'DEGs_plus_minus_others_specific_genes.csv')

## Compare top vs bottom
DEGs <- FindMarkers(sc.subtype.1, ident.1 = "top", ident.2 = "bottom", 
                   group.by = "Position", logfc.threshold = 0.1, 
                   min.pct = 0.1, slot = 'data')
write.csv(DEGs, file = 'DEGs_10layers_plus_vs_minus.csv')

DEGs <- FindMarkers(sc.subtype.1, ident.1 = "top", ident.2 = "bottom", 
                   min.pct = 0, logfc.threshold = 0)

DEGs_all <- DEGs
write.csv(DEGs_all, file = 'DEGs_10layers_plus_vs_minus_all.csv')

DEGs <- DEGs[DEGs$p_val_adj < 0.05, ]

## ============================================================================
## Part 6: Expression comparison plot
## ============================================================================

deg <- DEGs
deg$gene <- rownames(deg)

label_genes <- unique(c(
  "Ryr2","Hrc","Gja1","Fxyd1","Pdlim5","Nexn","Nppb","Txnip","Acsl1","Fndc5","Glul",
  "Tnnt2","Obscn","Perm1","Ckm","Arhgef3","Tns2","Bnip2","Neat1","Cst3",
  "Gja1","Ryr2","Hrc","Fxyd1","Nppb","Myh7","Pdlim5","Lmod2","Nexn","Dsp",
  "Itgb1","Cd36","Acsl1","Acadm","Acadl","Txnip","Fabp4",
  "Tnnt2","Arhgef3","Arhgef3","Obscn","Ckm","Perm1","Tnnt2","Neat1",
  "Ndufab1","Ndufb9","Ndufb10","Idh3b"
))

deg$label_color <- ifelse(deg$avg_log2FC > 0, "Up", "Down")

p <- ggplot(deg, aes(x = pct.1, y = pct.2)) +
  rasterise(
    geom_point(aes(color = avg_log2FC), size = 1, alpha = 1),
    dpi = 300
  ) +
  geom_point(
    data = subset(deg, gene %in% label_genes),
    aes(x = pct.1, y = pct.2),
    shape = 1,
    color = "black",
    stroke = 0.25,
    size = 1.2,
    inherit.aes = FALSE
  ) +
  geom_abline(
    slope = 1, intercept = 0,
    linetype = "dashed", color = "grey40", linewidth = 0.4
  ) +
  scale_color_gradient2(
    low = "#2166AC",
    mid = "grey90",
    high = "#B2182B",
    midpoint = 0,
    name = "Avg.log2FC",
    limits = c(-max(abs(deg$avg_log2FC), na.rm = TRUE),
               max(abs(deg$avg_log2FC), na.rm = TRUE))
  ) +
  ggnewscale::new_scale_color() +
  geom_text_repel(
    data = subset(deg, gene %in% label_genes),
    aes(label = gene, color = label_color),
    size = 2.5,
    max.overlaps = 50,
    segment.size = 0.2,
    segment.color = "grey50",
    show.legend = FALSE,
    min.segment.length = 0
  ) +
  scale_color_manual(
    values = c("Up" = "#B2182B", "Down" = "#2166AC"),
    guide = "none"
  ) +
  labs(
    x = "pct.1 (Plus expression)",
    y = "pct.2 (Minus expression)",
    title = "Gene Expression Comparison"
  ) +
  scale_x_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25)) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25)) +
  coord_fixed(ratio = 1) +
  theme_classic(base_size = 10) +
  theme(
    axis.line = element_line(linewidth = 0.4),
    axis.ticks = element_line(linewidth = 0.4),
    axis.text = element_text(color = "black", size = 8),
    axis.title = element_text(size = 9),
    plot.title = element_text(size = 10, hjust = 0.5),
    legend.position = "right",
    legend.key.height = unit(0.8, "cm"),
    legend.key.width  = unit(0.3, "cm"),
    legend.text  = element_text(size = 7),
    legend.title = element_text(size = 8),
    plot.margin = margin(10, 10, 10, 10)
  )

ggsave("F4c_Plus_Minus_expr_comparsion.pdf", p, width = 4.25, height = 3.75, units = "in")

## ============================================================================
## Part 7: Gene expression heatmap across slices
## ============================================================================

top_heart_genes <- c(
  "Actb","Eef1a2","Gapdh","Ppib","Rplp0","Tbp","Tuba1a",
  "Myh6","Des","Actc1","Tnni3",
  "Sdhb","Ryr2","Scn5a"
)

## Extract expression matrix
expr_mat <- GetAssayData(sc.subtype.1, slot = "data")

VlnPlot(sc.subtype.1, features = c("nCount_RNA", "nFeature_RNA"), 
        group.by = "Slices")

meta <- sc.subtype.1@meta.data[colnames(expr_mat), , drop = FALSE]

## Calculate mean expression per slice for each gene
gene_slice_expr <- list()

for (gene in top_heart_genes) {
  if (gene %in% rownames(expr_mat)) {
    gene_expr <- expr_mat[gene, ]
    tmp_df <- data.frame(
      gene = gene,
      expression = as.numeric(gene_expr),
      Slice = meta$Slices
    ) %>%
      group_by(Slice, gene) %>%
      summarise(mean_expr = mean(expression), .groups = "drop")
    
    gene_slice_expr[[gene]] <- tmp_df
  }
}

plot_data <- bind_rows(gene_slice_expr)

## Prepare data for plotting
slice_num <- as.integer(sub("^S", "", plot_data$Slice))
plot_data$Slice <- factor(plot_data$Slice,
                          levels = paste0("S", sort(unique(slice_num))))
plot_data$gene <- factor(plot_data$gene, levels = unique(plot_data$gene))

## Scale expression per gene
plot_data_scaled <- plot_data %>%
  group_by(gene) %>%
  mutate(scaled_expr = as.numeric(scale(mean_expr))) %>%
  ungroup() %>%
  mutate(gene = factor(gene, levels = rev(top_heart_genes)))

## Heatmap
p_heat <- ggplot(plot_data_scaled, aes(x = Slice, y = gene, fill = scaled_expr)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(
    low = "#4575b4", mid = "#ffffbf", high = "#d73027", midpoint = 0,
    name = "Scaled\nExpression"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5, color = "black"),
    axis.text.y = element_text(color = "black"),
    panel.grid = element_blank()
  ) +
  labs(x = "Slice", y = "Gene", title = "Row-Scaled Gene Expression Heatmap")

## Annotation bubble plot
ann <- results_df %>%
  filter(gene %in% top_heart_genes) %>%
  mutate(
    gene = factor(gene, levels = rev(top_heart_genes)),
    nlogFDR = -log10(padj + 1e-300),
    sizev = pmin(nlogFDR, 10)
  )

p_ann <- ggplot(ann, aes(x = 1, y = gene)) +
  geom_point(aes(size = sizev, color = slope), alpha = 0.95) +
  scale_color_gradient2(low = "#3B4CC0", mid = "grey95", high = "#B40426", 
                       name = "slope") +
  scale_size(range = c(0.5, 6), name = "-log10(FDR)") +
  scale_x_continuous(limits = c(0.5, 1.5), expand = c(0, 0)) +
  labs(x = NULL, y = NULL) +
  theme_void(base_size = 14) +
  theme(
    plot.margin = margin(5, 5, 5, 0),
    legend.position = "right"
  )

p_heat
