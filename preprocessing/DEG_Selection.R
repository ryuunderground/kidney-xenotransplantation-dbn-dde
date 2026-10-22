### NOISeq-sim DEG pipeline
### Criteria: prob(q) >= 0.95 AND |log2FC| >= 3

root <- "~/Documents/github/kidney-xenotransplantation-dbn-dde/preprocessing"
setwd(root)

# Packages
cran_packages <- c("clipr", "stringr", "calibrate", "ggplot2", "gridExtra", "ggrepel", "dplyr")
for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
bioc_packages <- c("edgeR", "NOISeq")
for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) BiocManager::install(pkg)
  library(pkg, character.only = TRUE)
}

# Sample days per survival group
Short_Samples <- c("1C", "3C", "5C", "7C", "14C", "28C", "35C", "42C", "49C")
Long_Samples <- c("1C", "7C", "14C", "21C", "34C", "42C", "49C", "56C", "63C", "70C", "77C", "84C", "91C", "98C", "105C", "112C")
Intermediate_Samples <- c("1C", "4C", "6C", "8C", "11C", "14C", "21C", "28C", "35C", "42C", "49C", "63C", "70C", "77C")

# Read a count file: "GeneID count" (space or tab separated)
read_count_file <- function(path) {
  x <- read.table(path, header = FALSE, sep = "", stringsAsFactors = FALSE)
  if (ncol(x) < 2) stop("Count file must have at least 2 columns: GeneID and count: ", path)
  colnames(x)[1:2] <- c("Gene", "Count")
  x$Gene <- as.character(x$Gene)
  x$Count <- as.numeric(x$Count)
  x
}

# NOISeq-sim parameters
pnr <- 0.2    # % of sequencing depth used per simulated replicate
nss <- 10     # number of simulated replicates
v   <- 0.02   # variability of replicate size
q_cut <- 0.95
lfc_cut <- 3

# Gene annotation table (Gene -> GeneName)
annotation_path <- file.path(root, "annotation/mac_ann.txt")
gene_annot <- read.csv(annotation_path, header = TRUE, stringsAsFactors = FALSE)
colnames(gene_annot)[1:2] <- c("Gene", "GeneName")

# Run once per date, swapping the sample list below for each survival group:
## 1M1, IMI: Short-term survival
## 5F1, VFI: Intermediate-term survival
## 2M1, IIMI: Long-term survival
for (date in Short_Samples) {  # 1M1 (IMI); use Intermediate_Samples / Long_Samples with the matching meta_folder for the other groups
  
  message("\n====================")
  message("Date: ", date)
  
  # 1) Load metadata for this date
  meta_folder <- "./inputs/1M1"
  meta_path <- file.path(meta_folder, paste0(date, ".csv"))
  meta_data <- read.csv(meta_path, header = TRUE, stringsAsFactors = FALSE)
  
  sample_names <- as.character(meta_data$Name)
  group <- as.factor(meta_data$Group)
  
  if ("Control" %in% levels(group)) group <- relevel(group, ref = "Control")
  
  print(table(group))
  if (length(levels(group)) != 2) {
    stop("NOISeq-sim DEG in this script assumes exactly 2 groups per date. Found: ",
         paste(levels(group), collapse = ", "))
  }
  
  # 2) Read counts and assemble the expression matrix
  expression_list <- vector("list", length(sample_names))
  gene_ids <- NULL
  
  for (i in seq_along(sample_names)) {
    file_path <- file.path(root, "Samples/1M1", sample_names[i])
    tmp <- read_count_file(file_path)
    
    if (is.null(gene_ids)) {
      gene_ids <- tmp$Gene
    } else {
      # align by GeneID in case row order differs between files
      if (!all(tmp$Gene == gene_ids)) {
        tmp <- tmp[match(gene_ids, tmp$Gene), ]
      }
    }
    expression_list[[i]] <- tmp$Count
  }
  
  counts <- do.call(cbind, expression_list)
  rownames(counts) <- gene_ids
  colnames(counts) <- sample_names
  counts <- as.matrix(counts)
  
  # 3) Basic filtering
  # (a) drop all-zero genes
  keep_nz <- rowSums(counts) > 0
  counts <- counts[keep_nz, , drop = FALSE]
  
  # (b) drop low-count genes via edgeR::filterByExpr
  dge_tmp <- edgeR::DGEList(counts = counts, group = group)
  keep_expr <- edgeR::filterByExpr(dge_tmp)
  counts <- counts[keep_expr, , drop = FALSE]
  message("Genes after filtering: ", nrow(counts))
  
  # 4) Build the NOISeq data object
  factors <- data.frame(Group = group)
  rownames(factors) <- colnames(counts)
  mydata <- NOISeq::readData(data = counts, factors = factors)
  
  # 5) Run NOISeq-sim (no biological replicates)
  res <- NOISeq::noiseq(
    mydata,
    factor = "Group",
    conditions = levels(group),
    k = NULL,
    norm = "tmm",
    pnr = pnr,
    nss = nss,
    v = v,
    lc = 0,
    replicates = "no"
  )
  
  # 6) Select DEGs by probability and log2FC cutoffs
  deg_df <- NOISeq::degenes(res, q = q_cut, M = NULL)
  deg_df <- deg_df[abs(deg_df$M) >= lfc_cut, , drop = FALSE]
  deg_df$Gene <- rownames(deg_df)
  
  # 7) Join gene annotation, drop unannotated genes
  deg_df <- dplyr::left_join(deg_df, gene_annot, by = "Gene")
  deg_df <- deg_df[!is.na(deg_df$GeneName) & deg_df$GeneName != "", ]
  
  # 8) Save DEG table and gene list
  out_dir <- file.path(root, "DEG_Results", "1M1", date)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  write.csv(deg_df, file.path(out_dir, paste0(date, "_NOISeqsim_q", q_cut, "_logFC", lfc_cut, ".csv")),
            row.names = FALSE, quote = FALSE)
  
  DEG_names <- unique(deg_df$GeneName)
  write.csv(DEG_names, file.path(out_dir, paste0(date, "_gene_list.csv")),
            row.names = FALSE, quote = FALSE)
  clipr::write_clip(DEG_names)
  
  message("DEG candidates (q>= ", q_cut, ", |log2FC|>= ", lfc_cut, "): ", length(DEG_names))
  
  # Tag direction and report up/down counts
  deg_df$Direction <- ifelse(deg_df$M > 0, "UP_reg", "DOWN_reg")
  n_up   <- sum(deg_df$Direction == "UP_reg")
  n_down <- sum(deg_df$Direction == "DOWN_reg")
  
  cat("up-regulated genes  :", n_up, "\n")
  cat("down-regulated genes:", n_down, "\n")
  cat("Total DEG:", nrow(deg_df), "\n")
  
  # 9) Plot M vs prob (prob is a probability, not a p-value)
  all_res <- res@results[[1]]
  all_res$Gene <- rownames(all_res)
  all_res <- dplyr::left_join(all_res, gene_annot, by = "Gene")
  
  all_res$group <- "NS"
  all_res$group[all_res$prob >= q_cut & all_res$M >=  lfc_cut] <- "UP"
  all_res$group[all_res$prob >= q_cut & all_res$M <= -lfc_cut] <- "DOWN"
  all_res$group <- factor(all_res$group, levels = c("UP", "DOWN", "NS"))
  
  p <- ggplot(all_res, aes(x = M, y = prob)) +
    geom_point(data = subset(all_res, group == "NS"),
               color = "gray80", size = 1.2, alpha = 0.5, show.legend = FALSE) +
    geom_point(data = subset(all_res, group != "NS"),
               aes(color = group), size = 1.8) +
    geom_vline(xintercept = c(-lfc_cut, lfc_cut), linetype = "dashed") +
    geom_hline(yintercept = q_cut, linetype = "dashed") +
    theme_bw() +
    labs(
      title = paste0("NOISeq-sim: ", date),
      x = "M (log2 Fold Change)",
      y = "prob (DE probability)"
    )
  
  ggsave(filename = file.path(out_dir, paste0(date, "_NOISeqsim_M_prob.pdf")),
         plot = p, width = 8, height = 6)
}
