# Figure 2: Seurat Analysis Workflow
rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(Seurat)
  library(ggplot2)
  library(DoubletFinder)
  library(tidyverse)
  library(rstatix)
})

# Color palette
mycol <- c(
  "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#8CD17D",
  "#B6992D", "#499894", "#E15759", "#FF9D9A", "#79706E",
  "#D37295", "#FABFD2", "#B07AA1", "#D4A6C8", "#9D7660",
  "#E58606", "#5D69B1", "#24796C", "#DAA51B", "#000000",
  "#99C945", "#ED645A", "#66C5CC", "#F6CF71", "#F89C74", 
  "#DCB0F2", "#87C55F", "#9EB9F3", "#FE88B1", "#C9DB74", 
  "#8BE0A4", "#B497E7", "#D3B484", "#B4D2BA", "#DE6E6E", 
  "#5A9E88", "#FF6F91", "#7C90F2", "#C3A38A", "#FFD121", "#088247"
)

# ========== Helper functions ==========
make_seurat_object_and_doublet_removal <- function(data_directory, project_name){
  colon.data <- read.csv(file = data_directory, header = TRUE, row.names = 1)
  currentSample <- CreateSeuratObject(counts = colon.data, project = project_name, 
                                      min.cells = 0, min.features = 0)
  currentSample[["percent.mt"]] <- PercentageFeatureSet(currentSample, pattern = "^mt-")
  
  # QC plots
  pdf(paste0("./result/qc_plots_", project_name, "_prefiltered.pdf"))
  print(VlnPlot(currentSample, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
                ncol = 3, pt.size = 0))
  dev.off()
  
  currentSample <- seurat_standard_normalize_and_scale(currentSample, FALSE)
  
  # DoubletFinder (for reference only, not filtered)
  nExp_poi <- round(0.5 * length(currentSample@meta.data$orig.ident) * 
                    length(currentSample@meta.data$orig.ident) / 10000)
  seu_colon <- doubletFinder_v3(currentSample, PCs = 1:20, pN = 0.25, pK = 0.09, 
                                nExp = nExp_poi, reuse.pANN = FALSE, sct = FALSE)
  
  seu_colon$doublet.class <- seu_colon[[paste0("DF.classifications_0.25_0.09_", nExp_poi)]]
  seu_colon[[paste0("DF.classifications_0.25_0.09_", nExp_poi)]] <- NULL
  pann <- grep(pattern = "^pANN", x = names(seu_colon@meta.data), value = TRUE)
  seu_colon$pANN <- seu_colon[[pann]]
  seu_colon[[pann]] <- NULL
  
  pdf(paste0("./result/UMAP_doublet_", project_name, ".pdf"))
  print(DimPlot(seu_colon, reduction = "umap", group.by = "doublet.class", 
                cols = c("#D51F26", "#272E6A")))
  dev.off()
  
  seu_colon <- DietSeurat(seu_colon, counts = TRUE, data = TRUE, 
                          scale.data = FALSE, assays = "RNA")
  return(seu_colon)
}

seurat_standard_normalize_and_scale <- function(colon, cluster, cluster_resolution = 0.8){
  colon <- NormalizeData(colon, normalization.method = "LogNormalize", scale.factor = 10000)
  colon <- FindVariableFeatures(colon, selection.method = "vst", nfeatures = 2000)
  all.genes <- rownames(colon)
  colon <- ScaleData(colon, features = all.genes)
  colon <- RunPCA(colon, features = VariableFeatures(object = colon), npcs = 20)
  if (cluster){
    colon <- FindNeighbors(colon, dims = 1:20)
    colon <- FindClusters(colon, resolution = cluster_resolution)
  }
  colon <- RunUMAP(colon, dims = 1:20)
  return(colon)
}

# ========== Data loading and preprocessing ==========
setwd('./01_gene_expression/3d_cell_matrices_symbol')
data_directory <- list.files()

samples <- c("Tile15", "Tile16",
             "Tile24", "Tile25", "Tile26", "Tile27", "Tile28",
             "Tile33", "Tile34", "Tile35", "Tile36", "Tile37", "Tile38",
             "Tile43", "Tile44", "Tile45", "Tile46", "Tile47", "Tile48", "Tile49",
             "Tile52", "Tile53", "Tile54", "Tile55", "Tile56", "Tile57", "Tile58",
             "Tile63", "Tile64", "Tile65", "Tile66", "Tile67", "Tile68",
             "Tile73", "Tile74", "Tile75", "Tile76", "Tile77", "Tile78",
             "Tile85", "Tile86")

# Process and merge samples
seu_list <- make_seurat_object_and_doublet_removal(data_directory[1], samples[1])

for (i in 2:length(samples)){
  sc.i <- make_seurat_object_and_doublet_removal(data_directory[i], samples[i])
  seu_list <- merge(seu_list, sc.i)
}

scRNA_harmony <- seu_list

# ========== Data integration and dimensionality reduction ==========
scRNA_harmony <- NormalizeData(scRNA_harmony) %>% 
  FindVariableFeatures() %>% 
  ScaleData() %>% 
  RunPCA(verbose = FALSE)

scRNA_harmony <- FindNeighbors(scRNA_harmony, dims = 1:10) %>% 
  FindClusters(resolution = 1.0) %>%
  RunUMAP(dims = 1:10)

Idents(scRNA_harmony) <- 'seurat_clusters'

# Add cell metadata
meta <- read.csv(file = 'cell_centroids.csv', stringsAsFactors = FALSE)
meta <- meta[, c(1, 2, 3)]
meta <- column_to_rownames(meta, var = 'track_id')
meta <- na.omit(meta)

# ========== Module-specific marker analysis ==========
setwd('/Volumes/T9/v4/expression/L1/09_module_sepecific_gene')

markers <- FindAllMarkers(scRNA_harmony, only.pos = TRUE, min.pct = 0, logfc.threshold = 0)
markers <- markers[markers$p_val_adj < 0.05, ]
write.csv(markers, "Module_specific_gene_expression.csv", row.names = FALSE)

# ========== Layer count distribution across modules ==========
setwd('/Volumes/T9/v4/expression/L1/14_layer_counts')

df <- scRNA_harmony@meta.data %>%
  dplyr::select(Module, layer_count) %>%
  filter(!is.na(Module), !is.na(layer_count)) %>%
  mutate(Module = as.factor(Module),
         layer_count = as.numeric(layer_count))

module_order <- df %>%
  count(Module, name = "n") %>%
  arrange(desc(n)) %>%
  pull(Module) %>%
  as.character()

df <- df %>% mutate(Module = factor(Module, levels = module_order))

# Statistical testing
kw <- df %>% kruskal_test(layer_count ~ Module)

wil <- df %>%
  pairwise_wilcox_test(layer_count ~ Module,
                       p.adjust.method = "BH",
                       exact = FALSE) %>%
  arrange(p.adj)

write.csv(wil, "Module_layerCount_wilcox_test.csv", row.names = FALSE)

# Visualization
p1 <- ggplot(df, aes(x = Module, y = layer_count)) +
  geom_boxplot(outlier.shape = NA, width = 0.7) +
  geom_jitter(width = 0.15, alpha = 0.25, size = 0.6) +
  theme_classic(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Layer count distribution across Modules",
       subtitle = paste0("Kruskal-Wallis p = ", signif(kw$p, 3)),
       x = "Module", y = "Layer count")

ggsave("Module_layerCount_boxplot.pdf", p1, width = 5, height = 4.8, dpi = 300)

df_prop <- df %>%
  mutate(layer_count_f = factor(layer_count, levels = sort(unique(layer_count)))) %>%
  count(Module, layer_count_f) %>%
  group_by(Module) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

p2 <- ggplot(df_prop, aes(x = Module, y = prop, fill = layer_count_f)) +
  geom_col(width = 0.8) +
  theme_classic(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(title = "Layer count composition within Modules",
       x = "Module", y = "Proportion", fill = "Layer count")

ggsave("Module_layerCount_stackedProportion.pdf", p2, width = 5, height = 4.8, dpi = 300)

# ========== Cell type annotation ==========
Idents(scRNA_harmony) <- "seurat_clusters"

scRNA_harmony <- RenameIdents(scRNA_harmony,
                              "0" = "Module_1", "1" = "Module_1", "2" = "Module_2", "3" = "Module_1",
                              "4" = "Module_1", "5" = "Module_1", "6" = "Module_1",
                              "7" = "Module_5", "8" = "Module_3", "9" = "Module_4", 
                              "10" = "Module_2", "11" = "Module_1",
                              "12" = "Module_6", "13" = "Module_6", "14" = "Others")

scRNA_harmony@meta.data$Module <- scRNA_harmony@active.ident
scRNA_harmony <- subset(scRNA_harmony, ident = c('Module_1', 'Module_2', 'Module_3', 'Module_4', 'Module_5', 'Module_6'))
write.csv(scRNA_harmony@meta.data, file = 'seurat_matadata.csv')

# Filter high-risk cells
scRNA_harmony <- subset(scRNA_harmony, cells = setdiff(colnames(scRNA_harmony), high_risk_cells))

# Merge metadata
meta <- read.csv(file = 'all_tiles_best_merged.csv', stringsAsFactors = FALSE)
meta <- column_to_rownames(meta, var = 'track_id')

scRNA_harmony@meta.data <- cbind(scRNA_harmony@meta.data, meta[rownames(scRNA_harmony@meta.data), ])
save(scRNA_harmony, file = 'VCMs_harmony_10per.Rda')

# ========== snRNA-seq reference data analysis ==========
setwd('/Volumes/T9/v4/expression/L1/public_snRNAseq_data')

library(scales) 

Idents(scRNA_harmony) <- 'celltype'

# Extract cell type subsets
scRNA_harmony_FBs <- subset(scRNA_harmony, ident = c('FBs'))
scRNA_harmony_Endo <- subset(scRNA_harmony, ident = c('Endo'))
scRNA_harmony_PCs <- subset(scRNA_harmony, ident = c('PCs'))
scRNA_harmony_Immu <- subset(scRNA_harmony, ident = c('MPs', 'Bcells', 'NKT'))
scRNA_harmony_CMs <- subset(scRNA_harmony, ident = c('CMs'))

scRNA_harmony_3dvcm <- scRNA_harmony
bc <- unique(trimws(readLines("region4_cells.txt", warn = FALSE)))
bc <- intersect(bc[nzchar(bc)], colnames(scRNA_harmony_3dvcm))
scRNA_harmony_3dvcm <- subset(scRNA_harmony_3dvcm, cells = bc)

# Define gene sets
cardiac_genes <- c("Ttn", "Ryr2", "Myh6") 

gene_sets <- list(
  FB = list(anchors = c("Pdgfra", "Abi3bp", "Bmper", "Ccbe1")),
  Endo = list(anchors = c("Vwf", "Plvap", "Clec1a")),
  PC = list(anchors = c("Pdgfrb", "Notch3", "Gucy1a2", "Arhgap42")),
  Immu = list(anchors = c("Dock2", "Lcp2", "Ms4a4a", "Hck", "Spi1"))
)

ref_objects <- list(
  FB = scRNA_harmony_FBs,
  Endo = scRNA_harmony_Endo,
  PC = scRNA_harmony_PCs,
  Immu = scRNA_harmony_Immu
)

obj_cm <- scRNA_harmony_CMs
obj_vcm <- scRNA_harmony_3dvcm  

# Helper function to extract expression matrix
GetExprMat <- function(obj, slot = "data", assay = NULL) {
  if (!is.null(assay)) DefaultAssay(obj) <- assay
  GetAssayData(obj, slot = slot)
}

# Calculate detection rate and mean expression for genes
calc_stats_for_group <- function(obj, genes, group_name, slot = "data", assay = NULL) {
  expr <- GetExprMat(obj, slot = slot, assay = assay)
  genes_present <- intersect(genes, rownames(expr))
  if (length(genes_present) == 0) {
    stop(sprintf("[%s] None of the genes found in expression matrix.", group_name))
  }
  sub <- expr[genes_present, , drop = FALSE]
  det_rate <- Matrix::rowMeans(sub > 0)
  mean_expr <- Matrix::rowMeans(sub)
  
  tibble(
    gene = genes_present,
    group = group_name,
    det_rate = as.numeric(det_rate),
    mean_expr = as.numeric(mean_expr)
  )
}

make_wide_one_set <- function(set_name, metric = c("det_rate", "mean_expr"),
                              gene_sets, ref_objects, obj_cm, obj_vcm,
                              slot = "data", assay = NULL) {
  metric <- match.arg(metric)
  anchors <- gene_sets[[set_name]]$anchors
  obj_ref <- ref_objects[[set_name]]
  
  df_ref <- calc_stats_for_group(obj_ref, anchors, paste0("snRNA-", set_name), slot, assay)
  df_cm  <- calc_stats_for_group(obj_cm, anchors, "snRNA-CM", slot, assay)
  df_vcm <- calc_stats_for_group(obj_vcm, anchors, "3D-VCM", slot, assay)
  
  bind_rows(df_ref, df_cm, df_vcm) %>%
    mutate(grp = case_when(
      group == "snRNA-CM" ~ "CM",
      group == "3D-VCM"   ~ "VCM",
      grepl("^snRNA-", group) ~ "REF",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(grp)) %>%
    select(gene, grp, value = .data[[metric]]) %>%
    pivot_wider(names_from = grp, values_from = value) %>%
    mutate(set = set_name)
}

theme_publication <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(
      axis.title = element_text(color = "black"),
      axis.text = element_text(color = "black"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold"),
      legend.position = "top",
      legend.title = element_text(face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "black", hjust = 0),
      panel.grid = element_blank()
    )
}

plot_carryover_panel <- function(
    target_set = c("FB", "Endo", "PC", "Immu"),
    cardiac_genes,
    gene_sets,
    ref_objects,
    obj_cm,
    obj_vcm,
    slot = "data",
    assay = NULL,
    clamp_w = TRUE
) {
  target_set <- match.arg(target_set)
  
  anchors <- gene_sets[[target_set]]$anchors
  if (is.null(anchors) || length(anchors) == 0) stop("anchors missing for target_set")
  
  obj_ref <- ref_objects[[target_set]]
  if (is.null(obj_ref)) stop("ref object missing for target_set")
  
  group_ref <- paste0("snRNA-", target_set)
  group_cm <- "snRNA-CM"
  group_vcm <- "3D-VCM"
  
  genes_all <- unique(c(cardiac_genes, anchors))
  
  df_ref <- calc_stats_for_group(obj_ref, genes_all, group_ref, slot, assay)
  df_cm <- calc_stats_for_group(obj_cm, genes_all, group_cm, slot, assay)
  df_vcm <- calc_stats_for_group(obj_vcm, genes_all, group_vcm, slot, assay)
  
  df <- bind_rows(df_ref, df_cm, df_vcm) %>%
    mutate(
      panel = case_when(
        gene %in% cardiac_genes ~ "Cardiomyocyte identity",
        gene %in% anchors ~ paste0(target_set, " identity (anchors)"),
        TRUE ~ "Other"
      )
    )
  
  df_w <- df %>%
    filter(gene %in% anchors) %>%
    select(gene, group, det_rate, panel) %>%
    tidyr::pivot_wider(names_from = group, values_from = det_rate) %>%
    mutate(
      w_raw = (`3D-VCM` - `snRNA-CM`) / (.data[[group_ref]] - `snRNA-CM`),
      w = if (clamp_w) pmin(pmax(w_raw, 0), 1) else w_raw,
      gene = factor(gene, levels = rev(anchors)),
      panel = factor(panel, levels = paste0(target_set, " identity (anchors)"))
    )
  
  vlines <- data.frame(
    xintercept = c(0, 0.5, 1),
    linetype = c("solid", "dashed", "solid"),
    linewidth = c(0.30, 0.25, 0.30),
    colour = c("grey70", "grey80", "grey70")
  )
  
  p_lollipop <- ggplot(df_w, aes(x = w, y = gene)) +
    geom_vline(data = vlines,
               aes(xintercept = xintercept, linetype = linetype, 
                   linewidth = linewidth, colour = colour),
               show.legend = FALSE) +
    scale_colour_identity() +
    geom_segment(aes(x = 0, xend = w, yend = gene), 
                 linewidth = 1.0, color = "grey85") +
    geom_point(shape = 21, size = 3.2, fill = "#3B82F6", 
               color = "black", stroke = 0.25) +
    facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
    scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1),
                      labels = c("CM-like", "Intermediate", paste0(target_set, "-like")),
                      expand = expansion(mult = c(0.02, 0.02))) +
    labs(x = sprintf("%s signal index w (based on detection rate)", target_set),
         y = NULL,
         title = sprintf("%s-associated signal in 3D-VCM cardiomyocytes", target_set),
         subtitle = sprintf("w = (3D-VCM - snRNA-CM) / (snRNA-%s - snRNA-CM)  %s",
                           target_set, if (clamp_w) "[clamped to 0–1]" else "")) +
    theme_publication(9)
  
  df_dot <- df %>%
    filter(gene %in% c(cardiac_genes, anchors)) %>%
    mutate(
      group = factor(group, levels = c(group_ref, "snRNA-CM", "3D-VCM")),
      gene = factor(gene, levels = c(cardiac_genes, anchors)),
      panel = factor(panel, levels = c("Cardiomyocyte identity", 
                                       paste0(target_set, " identity (anchors)")))
    )
  
  p_dot <- ggplot(df_dot, aes(x = group, y = gene)) +
    geom_point(aes(size = det_rate, color = mean_expr)) +
    facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
    scale_size_continuous(range = c(0.8, 5.5), breaks = c(0.1, 0.5, 0.9),
                          labels = percent_format(accuracy = 1)) +
    scale_color_gradient(low = "grey85", high = "black") +
    labs(x = NULL, y = NULL, size = "Detection rate", color = "Mean expr") +
    theme_publication(9) +
    theme(legend.position = "right", axis.text.x = element_text(angle = 0, hjust = 0.5))
  
  list(df = df, df_w = df_w, p_lollipop = p_lollipop, p_dot = p_dot)
}

# ========== Carryover analysis ==========
res_immu <- plot_carryover_panel("Immu", cardiac_genes, gene_sets, ref_objects, obj_cm, obj_vcm)
ggsave("Fig_ImmuCarryover_Dot.pdf", res_immu$p_dot, width = 3.6, height = 3.8, useDingbats = FALSE)

res_endo <- plot_carryover_panel("Endo", cardiac_genes, gene_sets, ref_objects, obj_cm, obj_vcm)
ggsave("Fig_Endo_Dot.pdf", res_endo$p_dot, width = 3.6, height = 3.8, useDingbats = FALSE)

res_pc <- plot_carryover_panel("PC", cardiac_genes, gene_sets, ref_objects, obj_cm, obj_vcm)
ggsave("Fig_PC_Dot.pdf", res_pc$p_dot, width = 3.6, height = 3.8, useDingbats = FALSE)

res_fb <- plot_carryover_panel("FB", cardiac_genes, gene_sets, ref_objects, obj_cm, obj_vcm)
ggsave("Fig_FB_Dot.pdf", res_fb$p_dot, width = 3.6, height = 3.8, useDingbats = FALSE)

# ========== Combined carryover index visualization ==========
df_w_all <- bind_rows(
  res_fb$df_w %>% mutate(set = "FB"),
  res_endo$df_w %>% mutate(set = "Endo"),
  res_pc$df_w %>% mutate(set = "PC"),
  res_immu$df_w %>% mutate(set = "Immu")
) %>%
  mutate(set = factor(set, levels = c("FB", "Endo", "PC", "Immu")),
         gene = as.character(gene))

vlines <- data.frame(
  xintercept = c(0, 0.5, 1),
  linetype = c("solid", "dashed", "solid"),
  linewidth = c(0.30, 0.25, 0.30),
  colour = c("grey70", "grey80", "grey70")
)

bands <- data.frame(
  xmin = c(0, 0.5),
  xmax = c(0.5, 1),
  fill = c("#EFF6FF", "#ECFDF5")
)

p_all_lollipop <- ggplot(df_w_all, aes(x = w, y = gene)) +
  geom_rect(data = bands,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = fill),
            inherit.aes = FALSE, colour = NA) +
  scale_fill_identity() +
  geom_vline(data = vlines,
             aes(xintercept = xintercept, linetype = linetype, 
                 linewidth = linewidth, colour = colour),
             show.legend = FALSE) +
  scale_colour_identity() +
  geom_segment(aes(x = 0, xend = w, yend = gene), 
               linewidth = 1.0, color = "grey85") +
  geom_point(shape = 21, size = 3.0, fill = "#3B82F6", 
             color = "black", stroke = 0.25) +
  facet_grid(set ~ ., scales = "free_y", space = "free_y") +
  scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1),
                    labels = c("CM-like", "Intermediate", "Ref-like"),
                    expand = expansion(mult = c(0.02, 0.02))) +
  labs(x = "Signal index w (based on detection rate)", y = NULL,
       title = "Carryover index (w) across 4 reference lineages") +
  theme_publication(9) +
  theme(panel.background = element_rect(fill = NA, colour = NA),
        plot.background = element_rect(fill = "white", colour = NA))

ggsave("Fig_All4_Anchors_DetRate_facet.pdf", p_all_lollipop,
       width = 3, height = 4, useDingbats = FALSE)

# ========== Detection rate and expression comparison ==========
get_cell_expr <- function(obj, genes, group_name, slot = "data", assay = NULL) {
  expr <- GetExprMat(obj, slot, assay)
  genes_present <- intersect(genes, rownames(expr))
  if (length(genes_present) == 0) {
    return(tibble(gene = character(), group = character(), 
                  cell = character(), expr = numeric()))
  }
  
  sub <- expr[genes_present, , drop = FALSE]
  as.matrix(sub) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("gene") %>%
    pivot_longer(-gene, names_to = "cell", values_to = "expr") %>%
    mutate(group = group_name)
}

calc_pvalues <- function(set_name, gene_sets, obj_cm, obj_vcm, 
                         slot = "data", assay = NULL) {
  anchors <- gene_sets[[set_name]]$anchors
  
  df_cm <- get_cell_expr(obj_cm, anchors, "CM", slot, assay)
  df_vcm <- get_cell_expr(obj_vcm, anchors, "VCM", slot, assay)
  
  bind_rows(df_cm, df_vcm) %>%
    group_by(gene) %>%
    summarise(p_value = wilcox.test(expr[group == "CM"], 
                                    expr[group == "VCM"])$p.value,
              .groups = "drop") %>%
    mutate(set = set_name,
           p_label = case_when(
             p_value < 0.001 ~ "***",
             p_value < 0.01 ~ "**",
             p_value < 0.05 ~ "*",
             TRUE ~ "ns"
           ))
}

make_wide_one_set_with_pval <- function(set_name, metric = c("det_rate", "mean_expr"),
                                        gene_sets, ref_objects, obj_cm, obj_vcm,
                                        slot = "data", assay = NULL) {
  metric <- match.arg(metric)
  anchors <- gene_sets[[set_name]]$anchors
  if (is.null(anchors) || length(anchors) == 0) stop("anchors missing: ", set_name)
  
  obj_ref <- ref_objects[[set_name]]
  if (is.null(obj_ref)) stop("ref object missing: ", set_name)
  
  df_ref <- calc_stats_for_group(obj_ref, anchors, paste0("snRNA-", set_name), slot, assay)
  df_cm <- calc_stats_for_group(obj_cm, anchors, "snRNA-CM", slot, assay)
  df_vcm <- calc_stats_for_group(obj_vcm, anchors, "3D-VCM", slot, assay)
  
  df_wide <- bind_rows(df_ref, df_cm, df_vcm) %>%
    mutate(grp = case_when(
      group == "snRNA-CM" ~ "CM",
      group == "3D-VCM" ~ "VCM",
      grepl("^snRNA-", group) ~ "REF",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(grp)) %>%
    select(gene, grp, value = all_of(metric)) %>%
    group_by(gene, grp) %>%
    summarise(value = mean(value), .groups = "drop") %>%
    pivot_wider(names_from = grp, values_from = value) %>%
    mutate(set = set_name)
  
  df_pval <- calc_pvalues(set_name, gene_sets, obj_cm, obj_vcm, slot, assay)
  df_wide %>% left_join(df_pval, by = c("gene", "set"))
}

plot_cm_vcm_ref_with_pval <- function(df_wide, xlab, title,
                                      x_scale = c("percent01", "continuous"),
                                      add_bands = FALSE,
                                      show_pval = FALSE) {
  x_scale <- match.arg(x_scale)
  
  df_plot <- df_wide %>%
    mutate(set = factor(set, levels = c("FB", "Endo", "PC", "Immu")),
           gene = as.character(gene)) %>%
    group_by(set) %>%
    arrange(desc(REF), .by_group = TRUE) %>%
    mutate(gene_f = factor(gene, levels = rev(unique(gene)))) %>%
    ungroup()
  
  if (show_pval && "p_label" %in% colnames(df_plot)) {
    x_max <- max(c(df_plot$CM, df_plot$VCM, df_plot$REF), na.rm = TRUE)
    df_plot <- df_plot %>%
      mutate(x_pval = pmax(CM, VCM, REF, na.rm = TRUE) + 0.05 * x_max)
  }
  
  p <- ggplot(df_plot, aes(y = gene_f))
  
  if (add_bands) {
    bands_df <- data.frame(xmin = c(0, 0.5), xmax = c(0.5, 1),
                           fill = c("#EFF6FF", "#ECFDF5"))
    p <- p +
      geom_rect(data = bands_df,
                aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = fill),
                inherit.aes = FALSE, colour = NA, show.legend = FALSE) +
      scale_fill_identity()
  }
  
  p <- p +
    geom_segment(aes(x = CM, xend = REF, yend = gene_f),
                 linewidth = 1.2, colour = "grey85") +
    geom_point(aes(x = CM), shape = 21, size = 2.6, fill = "grey55",
               colour = "black", stroke = 0.2) +
    geom_point(aes(x = REF), shape = 21, size = 2.6, fill = "grey25",
               colour = "black", stroke = 0.2) +
    geom_point(aes(x = VCM), shape = 21, size = 3.2, fill = "#2563EB",
               colour = "black", stroke = 0.25)
  
  if (show_pval && "p_label" %in% colnames(df_plot)) {
    p <- p +
      geom_text(data = df_plot,
                aes(x = x_pval, y = gene_f, label = p_label),
                size = 2.5, hjust = 0, color = "black")
  }
  
  p <- p +
    facet_grid(set ~ ., scales = "free_y", space = "free_y") +
    labs(x = xlab, y = NULL, title = title) +
    theme_publication(9) +
    theme(panel.background = element_rect(fill = NA, colour = NA))
  
  if (x_scale == "percent01") {
    p <- p + scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                                labels = percent_format(accuracy = 1))
  } else {
    p <- p + scale_x_continuous(labels = label_number(accuracy = 0.01),
                                expand = expansion(mult = c(0.02, 0.15)))
  }
  p
}

df_det <- bind_rows(
  make_wide_one_set("FB", "det_rate", gene_sets, ref_objects, obj_cm, obj_vcm),
  make_wide_one_set("Endo", "det_rate", gene_sets, ref_objects, obj_cm, obj_vcm),
  make_wide_one_set("PC", "det_rate", gene_sets, ref_objects, obj_cm, obj_vcm),
  make_wide_one_set("Immu", "det_rate", gene_sets, ref_objects, obj_cm, obj_vcm)
)

p_prop <- plot_cm_vcm_ref_with_pval(df_det,
  xlab = "Detection rate (>0)",
  title = "Detection-rate comparison across 4 reference lineages",
  x_scale = "percent01", add_bands = TRUE, show_pval = FALSE)

df_mean_with_pval <- bind_rows(
  make_wide_one_set_with_pval("FB", "mean_expr", gene_sets, ref_objects, obj_cm, obj_vcm),
  make_wide_one_set_with_pval("Endo", "mean_expr", gene_sets, ref_objects, obj_cm, obj_vcm),
  make_wide_one_set_with_pval("PC", "mean_expr", gene_sets, ref_objects, obj_cm, obj_vcm),
  make_wide_one_set_with_pval("Immu", "mean_expr", gene_sets, ref_objects, obj_cm, obj_vcm)
)

p_mean <- plot_cm_vcm_ref_with_pval(df_mean_with_pval,
  xlab = "Mean expression",
  title = "Mean-expression comparison across 4 reference lineages",
  x_scale = "continuous", add_bands = FALSE, show_pval = TRUE)

ggsave("Fig_All4_Anchors_DetRate_0to100_facet.pdf", p_prop, width = 3, height = 4, useDingbats = FALSE)
ggsave("Fig_All4_Anchors_MeanExpr_facet.pdf", p_mean, width = 3, height = 4, useDingbats = FALSE)

# ========== Module score analysis ==========
obj <- obj_vcm

# Calculate module scores for each signature
sig_list <- list(
  CM = intersect(cardiac_genes, rownames(obj)),
  FB = intersect(gene_sets$FB$anchors, rownames(obj)),
  Endo = intersect(gene_sets$Endo$anchors, rownames(obj)),
  PC = intersect(gene_sets$PC$anchors, rownames(obj)),
  Immu = intersect(gene_sets$Immu$anchors, rownames(obj))
)

for (nm in names(sig_list)) {
  obj <- AddModuleScore(obj, features = list(sig_list[[nm]]), 
                        name = paste0("Score_", nm))
}

md <- obj@meta.data %>%
  transmute(CM = .data[["Score_CM1"]],
            FB = .data[["Score_FB1"]],
            Endo = .data[["Score_Endo1"]],
            PC = .data[["Score_PC1"]],
            Immu = .data[["Score_Immu1"]])

theme_density_custom <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(axis.title = element_text(color = "black"),
          axis.text = element_text(color = "black"),
          plot.title = element_text(face = "bold", hjust = 0),
          plot.subtitle = element_text(color = "black", hjust = 0),
          legend.position = "top",
          legend.title = element_blank(),
          panel.grid = element_blank())
}

plot_two_density <- function(md, other = c("FB", "Endo", "PC", "Immu"),
                             xlim = NULL, q_high = 0.90) {
  other <- match.arg(other)
  
  df <- bind_rows(
    tibble(score = md$CM, type = "CM"),
    tibble(score = md[[other]], type = other)
  ) %>% filter(is.finite(score))
  
  if (is.null(xlim)) {
    xlim <- as.numeric(quantile(df$score, probs = c(0.01, 0.99), na.rm = TRUE))
  }
  
  med_cm <- median(md$CM, na.rm = TRUE)
  med_ot <- median(md[[other]], na.rm = TRUE)
  thr_ot <- quantile(md[[other]], q_high, na.rm = TRUE)
  frac_high <- mean(md[[other]] >= thr_ot, na.rm = TRUE)
  
  cols <- c("CM" = "#111827", other = "#2563EB")
  cols[[other]] <- cols[["other"]]
  
  ggplot(df, aes(x = score, colour = type, fill = type)) +
    geom_density(linewidth = 0.5, alpha = 0.22, adjust = 1) +
    geom_vline(xintercept = med_cm, colour = cols[["CM"]], 
               linewidth = 0.6, linetype = "solid") +
    geom_vline(xintercept = med_ot, colour = cols[[other]], 
               linewidth = 0.6, linetype = "solid") +
    scale_colour_manual(values = cols) +
    scale_fill_manual(values = cols) +
    scale_x_continuous(limits = xlim, expand = expansion(mult = c(0.01, 0.01))) +
    labs(x = "Module score", y = "Density",
         title = paste0("3D-VCM: CM vs ", other, " score distribution"),
         subtitle = sprintf("%s-high (top %.0f%%) = %.1f%%",
                           other, (1 - q_high) * 100, frac_high * 100)) +
    theme_density_custom(9) +
    theme(legend.margin = margin(b = 2))
}

# Plot density distributions
all_scores <- c(md$CM, md$FB, md$Endo, md$PC, md$Immu)
xlim_global <- as.numeric(quantile(all_scores, c(0.01, 0.99), na.rm = TRUE))

p_fb <- plot_two_density(md, "FB", xlim = xlim_global, q_high = 0.90)
p_endo <- plot_two_density(md, "Endo", xlim = xlim_global, q_high = 0.90)
p_pc <- plot_two_density(md, "PC", xlim = xlim_global, q_high = 0.90)
p_immu <- plot_two_density(md, "Immu", xlim = xlim_global, q_high = 0.90)

ggsave("Fig_Density_CM_vs_FB.pdf", p_fb, width = 2.4, height = 2.4, useDingbats = FALSE)
ggsave("Fig_Density_CM_vs_Endo.pdf", p_endo, width = 2.4, height = 2.4, useDingbats = FALSE)
ggsave("Fig_Density_CM_vs_PC.pdf", p_pc, width = 2.4, height = 2.4, useDingbats = FALSE)
ggsave("Fig_Density_CM_vs_Immu.pdf", p_immu, width = 2.4, height = 2.4, useDingbats = FALSE)


# sessionInfo()
# R version 4.1.3 (2022-03-10)
# Platform: x86_64-apple-darwin17.0 (64-bit)
# Running under: macOS 14.1
# 
# Matrix products: default
# LAPACK: /Library/Frameworks/R.framework/Versions/4.1/Resources/lib/libRlapack.dylib
# 
# locale:
# [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
# 
# attached base packages:
# [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# other attached packages:
# [1] rstatix_0.7.2       lubridate_1.9.3     forcats_1.0.0       stringr_1.5.1       purrr_1.0.2        
# [6] readr_2.1.5         tidyr_1.3.1         tibble_3.2.1        tidyverse_2.0.0     DoubletFinder_2.0.3
# [11] ggplot2_3.5.1       SeuratObject_4.1.3  Seurat_4.3.0        dplyr_1.1.4        
# 
# loaded via a namespace (and not attached):
# [1] Rtsne_0.16             colorspace_2.1-0       deldir_1.0-6           ggridges_0.5.6        
# [5] rstudioapi_0.16.0      spatstat.data_3.0-4    leiden_0.4.3.1         listenv_0.9.1         
# [9] ggrepel_0.9.5          fansi_1.0.6            codetools_0.2-20       splines_4.1.3         
# [13] polyclip_1.10-6        jsonlite_1.8.8         broom_1.0.6            ica_1.0-3             
# [17] cluster_2.1.4          png_0.1-8              uwot_0.1.14            shiny_1.8.1.1         
# [21] sctransform_0.3.5      spatstat.sparse_3.0-1  compiler_4.1.3         httr_1.4.7            
# [25] backports_1.4.1        Matrix_1.5-1           fastmap_1.1.1          lazyeval_0.2.2        
# [29] cli_3.6.1              later_1.3.2            htmltools_0.5.8.1      tools_4.1.3           
# [33] igraph_1.4.2           gtable_0.3.5           glue_1.8.0             RANN_2.6.2            
# [37] reshape2_1.4.4         Rcpp_1.0.14            carData_3.0-5          scattermore_1.2       
# [41] vctrs_0.6.5            spatstat.explore_3.1-0 nlme_3.1-162           progressr_0.14.0      
# [45] lmtest_0.9-40          spatstat.random_3.1-4  globals_0.16.3         timechange_0.3.0      
# [49] mime_0.12              miniUI_0.1.1.1         lifecycle_1.0.4        irlba_2.3.5.1         
# [53] goftest_1.2-3          future_1.33.2          MASS_7.3-60            zoo_1.8-12            
# [57] scales_1.3.0           hms_1.1.3              promises_1.2.0.1       spatstat.utils_3.0-3  
# [61] parallel_4.1.3         RColorBrewer_1.1-3     reticulate_1.40.0      pbapply_1.7-2         
# [65] gridExtra_2.3          stringi_1.7.12         rlang_1.1.4            pkgconfig_2.0.3       
# [69] matrixStats_0.63.0     lattice_0.21-8         ROCR_1.0-11            tensor_1.5            
# [73] patchwork_1.2.0        htmlwidgets_1.6.4      cowplot_1.1.3          tidyselect_1.2.1      
# [77] parallelly_1.37.1      RcppAnnoy_0.0.20       plyr_1.8.9             magrittr_2.0.3        
# [81] R6_2.5.1               generics_0.1.3         DBI_1.2.3              pillar_1.9.0          
# [85] withr_3.0.0            fitdistrplus_1.1-11    survival_3.5-5         abind_1.4-5           
# [89] sp_1.6-0               future.apply_1.11.2    car_3.1-2              KernSmooth_2.23-21    
# [93] utf8_1.2.4             spatstat.geom_3.2-1    plotly_4.10.4          tzdb_0.4.0            
# [97] grid_4.1.3             data.table_1.14.8      digest_0.6.31          xtable_1.8-4          
# [101] httpuv_1.6.10          munsell_0.5.1          viridisLite_0.4.2   
