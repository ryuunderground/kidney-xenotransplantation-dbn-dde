### Build DEG candidate expression table
### Repeat 3 times, changing the sample folder/output names:
## 1M1: Short-term survival
## 5F1: Intermediate-term survival
## 2M1: Long-term survival

root <- "~/Documents/github/kidney-xenotransplantation-dbn-dde/preprocessing"
setwd(root)
getwd()

library(dplyr)
library(readr)

# Step 1: locate all NOISeq gene_list.csv files for this survival group
deg_dir <- "DEG_Results/1M1"
deg_files <- list.files(
  path = deg_dir,
  pattern = "_gene_list\\.csv$",
  full.names = TRUE,
  recursive = TRUE
)

# Step 2: merge into one deduplicated DEG list
deg_union <- deg_files %>%
  lapply(function(f) {
    df <- read_csv(f, col_names = TRUE)
    colnames(df) <- "GeneName"
    df
  }) %>%
  bind_rows() %>%
  distinct()

write_csv(deg_union, "./total_deg_lists/short_total_deg_lists.csv")
cat("Total unique DEG candidates:", nrow(deg_union), "\n")

# Step 3: load log-TMM expression matrix, keep only DEG genes
expr <- read.csv("./TMM_CSV/1M1/short_logTMM_geneName.csv",
                 stringsAsFactors = FALSE, check.names = FALSE)
colnames(expr)[1] <- "Gene"

merged_data <- expr %>%
  dplyr::inner_join(
    deg_union %>% dplyr::mutate(Gene = GeneName),
    by = "Gene"
  ) %>%
  dplyr::filter(!is.na(GeneName) & GeneName != "") %>%
  dplyr::select(-Gene) %>%
  dplyr::relocate(GeneName, .before = 1)

expression_cols <- setdiff(colnames(merged_data), "GeneName")

# Step 4: save
write.csv(merged_data, "./DBN-DDE_candidates/short_candidates_data.csv", row.names = FALSE)

