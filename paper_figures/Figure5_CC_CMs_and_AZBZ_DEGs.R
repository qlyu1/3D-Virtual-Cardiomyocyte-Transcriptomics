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
library(ggrastr)

## Color palette
mycol <- c("#4E79A7","#A0CBE8","#F28E2B","#FFBE7D","#8CD17D",
           "#B6992D","#499894","#E15759","#FF9D9A","#79706E",
           "#D37295","#FABFD2","#B07AA1","#D4A6C8","#9D7660",
           "#E58606","#5D69B1","#24796C",'#DAA51B','#000000',
           '#99C945','#ED645A')

setwd('/Volumes/T9/v4/expression-0121/04_module_plot')

## Differential expression analysis: Border Zone vs Adjacent Zone
Idents(scRNA_harmony) <- 'Module'
scRNA_harmony <- RenameIdents(scRNA_harmony, `Module_2` = "BZ", `Module_4` = "AZ")

deg <- FindMarkers(scRNA_harmony, ident.1 = "BZ", ident.2 = "AZ", 
                   logfc.threshold = 0, min.pct = 0)
deg <- deg[deg$p_val_adj < 0.05, ]
write.csv(deg, file = 'DEGs_M2_vs_M4_all.csv')

## Calculate mean UMI counts per group
genes <- rownames(deg)
cells_m2 <- WhichCells(scRNA_harmony, idents = "BZ")
cells_m4 <- WhichCells(scRNA_harmony, idents = "AZ")

mat_counts <- GetAssayData(scRNA_harmony, assay = "RNA", slot = "counts")[genes, , drop = FALSE]
m2_counts <- mat_counts[, cells_m2, drop = FALSE]
m4_counts <- mat_counts[, cells_m4, drop = FALSE]

deg$mean_umi_m2 <- Matrix::rowMeans(m2_counts)[genes]
deg$mean_umi_m4 <- Matrix::rowMeans(m4_counts)[genes]
deg$mean_umi_max <- pmax(deg$mean_umi_m2, deg$mean_umi_m4)
deg$mean_umi_min <- pmin(deg$mean_umi_m2, deg$mean_umi_m4)

## Select genes for labeling
label_genes <- c(
  "Hmox1","Lgals3","Spp1","Timp1","Lox","Loxl2","Fstl1","Ccl2",
  "Ccl7","Serpina3n","Saa3","Anxa1","Tnc","Fn1","Rcan1",
  "Kcnip2","Abcc9","Kcnj3","Plin5","Perm1","Ogdhl","Acacb",
  "Etfdh","Bckdha","Bnip3","Mylk4","Lrrc39")

deg$label_color <- ifelse(deg$avg_log2FC > 0, "Up", "Down")

## Plot expression comparison
ggplot(deg, aes(x = pct.1, y = pct.2)) +
  rasterise(
    geom_point(aes(color = avg_log2FC), size = 0.5, alpha = 1),
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
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", 
              color = "grey40", linewidth = 0.4) +
  scale_color_gradient2(
    low = "#2166AC",
    mid = "grey90",
    high = "#B2182B",
    midpoint = 0,
    name = "Avg.log2FC",
    limits = c(-max(abs(deg$avg_log2FC)), max(abs(deg$avg_log2FC)))
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
    x = "pct.1 (Border zone expression)",
    y = "pct.2 (Adjacent zone expression)",
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
    legend.key.width = unit(0.3, "cm"),
    legend.text = element_text(size = 7),
    legend.title = element_text(size = 8),
    plot.margin = margin(10, 10, 10, 10)
  )

ggsave("expression_comparison.pdf", width = 4, height = 3.5, units = "in")

## ============================================================================
## Part 2: Proliferating cell analysis
## ============================================================================

rm(list = ls())

setwd('/Volumes/T9/v4/expression-0121/07_proliferating_cell_inM2M4/Proliferating_Score_F4e')

## Subset proliferating cells
barcodes <- readLines("./proliferating_cells.txt")
sc.subtype.1 <- subset(scRNA_harmony, cells = barcodes)
sc.subtype.1@meta.data$Group <- 'CC_CMs'

## Calculate proliferation score
prolif_genes <- c("Aurkb", "Mki67", "Top2a")
prolif_genes <- prolif_genes[prolif_genes %in% rownames(sc.subtype.1)]

expr_matrix <- GetAssayData(sc.subtype.1, slot = "data", assay = "RNA")
ProScore <- colSums(expr_matrix[prolif_genes, , drop = FALSE])
sc.subtype.1$ProScore <- ProScore

## Compute gene-wise correlation with proliferation score
expr_matrix_full <- as.matrix(expr_matrix)

cor_results <- apply(expr_matrix_full, 1, function(gene_expr) {
  cor(gene_expr, ProScore, method = "pearson")
})

avg_expr <- rowMeans(expr_matrix_full)

cor_df <- data.frame(
  Gene = names(cor_results),
  Correlation = cor_results,
  AvgExpression = avg_expr,
  stringsAsFactors = FALSE
)

cor_df <- cor_df[!is.na(cor_df$Correlation), ]
write.csv(cor_df, file = 'ProScore_cor_Genes.csv')

## Define genes for labeling
pos15 <- c(
  "Mki67","Top2a","Cip2a","Tacc3","Kif20b","Aspm","Mcmbp","Lmnb2","E2f4","Npm1",
  "Usp45","Exo5","Chmp4b","Myh9","Actg1"
)

neg15 <- c(
  "Tnni3","Myh7","Myh6","Ttn","Atp2a2","Fabp3","Cox7b","Uqcrfs1","Ndufs3","Ndufa13",
  "Atp5b","Pdhb","Mdh2","Fhl2","Actn2"
)

cor_df <- cor_df %>%
  mutate(
    Label = case_when(
      Gene %in% pos15 ~ Gene,
      Gene %in% neg15 ~ Gene,
      TRUE ~ ""
    ),
    LabelGroup = case_when(
      Gene %in% pos15 ~ "Positive",
      Gene %in% neg15 ~ "Negative",
      TRUE ~ "None"
    )
  )

label_df <- cor_df %>% filter(Label != "")

## Plot theme used across the figure
theme_publication <- function() {
  theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.text = element_text(size = 10, color = "black"),
      axis.title = element_text(size = 11, face = "bold"),
      legend.position = "right",
      legend.background = element_rect(fill = "white", color = NA),
      legend.key.size = unit(0.4, "cm"),
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 9),
      plot.margin = margin(10, 10, 10, 10)
    )
}

## Calculate plot limits
y_min <- floor(min(cor_df$Correlation, na.rm = TRUE) * 10) / 10 - 0.1
y_max <- ceiling(max(cor_df$Correlation, na.rm = TRUE) * 10) / 10 + 0.1

## Generate correlation plot
p <- ggplot(cor_df, aes(x = AvgExpression, y = Correlation)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.3) +
  rasterise(
    geom_point(aes(color = Correlation), size = 1.5, alpha = 0.8),
    dpi = 600
  ) +
  scale_color_gradient2(
    low = "#2166AC",
    mid = "#F7F7F7",
    high = "#B2182B",
    midpoint = 0,
    limits = c(y_min, y_max),
    oob = scales::squish,
    name = "Pearson\nCorrelation"
  ) +
  geom_point(
    data = label_df,
    aes(x = AvgExpression, y = Correlation, fill = Correlation),
    shape = 21,
    size = 2.2,
    stroke = 0.35,
    color = "black",
    alpha = 1,
    show.legend = FALSE
  ) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "#F7F7F7",
    high = "#B2182B",
    midpoint = 0,
    limits = c(y_min, y_max),
    oob = scales::squish,
    guide = "none"
  ) +
  geom_text_repel(
    data = subset(label_df, LabelGroup == "Positive"),
    aes(label = Label),
    color = "#B2182B",
    size = 3,
    fontface = "italic",
    segment.color = "#B2182B",
    segment.size = 0.3,
    segment.curvature = -0.2,
    segment.ncp = 3,
    segment.angle = 20,
    box.padding = 0.5,
    point.padding = 0.3,
    max.overlaps = 30,
    min.segment.length = 0,
    force = 2,
    force_pull = 0.5,
    show.legend = FALSE
  ) +
  geom_text_repel(
    data = subset(label_df, LabelGroup == "Negative"),
    aes(label = Label),
    color = "#2166AC",
    size = 3,
    fontface = "italic",
    segment.color = "#2166AC",
    segment.size = 0.3,
    segment.curvature = -0.2,
    segment.ncp = 3,
    segment.angle = 20,
    box.padding = 0.5,
    point.padding = 0.3,
    max.overlaps = 30,
    min.segment.length = 0,
    force = 2,
    force_pull = 0.5,
    show.legend = FALSE
  ) +
  labs(
    x = "Average Expression",
    y = "Correlation with Proliferating",
    title = "Gene-wise Pearson Correlation with Proliferating"
  ) +
  theme_publication() +
  coord_cartesian(ylim = c(y_min, y_max)) +
  scale_y_continuous(breaks = seq(round(y_min, 1), round(y_max, 1), 0.2)) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.05)))

ggsave("Gene_Correlation_with_Proliferation.pdf", 
       plot = p, width = 5, height = 4, device = cairo_pdf)




# Plot gene expression distribution in 3D-VCMs 
cellcycle <- c("Aurka","Aurkb","Foxo1","Mki67","Nrg1","Pcna","Cdk1","Cdk2",
               "Ccna2","Top2a","Kif23")

mat <- GetAssayData(scRNA_harmony, assay = "RNA", slot = "data")
genes_use <- intersect(cellcycle, rownames(mat))

expr_matrix <- mat[genes_use, , drop = FALSE]
write.csv(as.data.frame(as.matrix(expr_matrix)),file = "genes_normalized_matrix_plot.csv")
# Downstream: visualize this matrix in Jupyter Notebook
