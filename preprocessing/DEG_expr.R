### DBN-DDE candidate gene filtering pipeline
### Each scale (short/inter/long) is filtered independently by its own
### variance / lag-1 autocorrelation cutoff — no cross-scale gene overlap required.

library(dplyr)

root <- "~/Documents/github/kidney-xenotransplantation-dbn-dde/preprocessing/DBN-DDE_candidates/"
setwd(root)

candidate_files <- c(
  short = "./short_candidates_data_edited.csv",
  inter = "./inter_candidates_data_edited.csv",
  long  = "./long_candidates_data_edited.csv"
)

output_files <- c(
  short = "../DEG_expr/short_expr.csv",
  inter = "../DEG_expr/inter_expr.csv",
  long  = "../DEG_expr/long_expr.csv"
)

var_quantile <- 0.3
acf_quantile <- 0.3

# Step 1: compute variance and lag-1 autocorrelation per gene
compute_var_acf <- function(expr) {
  var_values <- apply(expr, 1, var, na.rm = TRUE)
  
  acf_values <- apply(expr, 1, function(x) {
    if (length(x) < 3) return(NA)
    x1 <- x[-1]
    x2 <- x[-length(x)]
    if (sd(x1) == 0 || sd(x2) == 0) return(NA)
    cor(x1, x2, use = "complete.obs")
  })
  
  list(var = var_values, acf = abs(acf_values))
}

# Step 2: filter one candidate file down to genes above its own variance/ACF quantile cutoffs
filter_by_variability <- function(input_path, output_path, var_quantile, acf_quantile) {
  candidate_data <- read.csv(
    input_path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE
  )
  
  if (!"GeneName" %in% colnames(candidate_data)) {
    stop(paste0(input_path, " has no GeneName column."))
  }
  
  candidate_data$GeneName <- trimws(as.character(candidate_data$GeneName))
  
  gene_names <- candidate_data$GeneName
  expr <- candidate_data %>%
    select(-GeneName, -any_of("Control")) %>%
    as.matrix()
  rownames(expr) <- gene_names
  
  stats <- compute_var_acf(expr)
  var_values <- stats$var
  acf_values <- stats$acf
  acf_values[is.na(acf_values)] <- 0
  
  var_threshold <- quantile(var_values, var_quantile, na.rm = TRUE)
  acf_threshold <- quantile(acf_values, acf_quantile, na.rm = TRUE)
  
  keep <- var_values > var_threshold & acf_values > acf_threshold
  filtered_data <- candidate_data[keep, , drop = FALSE]
  
  write.csv(filtered_data, output_path, row.names = FALSE, quote = FALSE)
  
  cat("\nInput file:", input_path, "\n")
  cat("Saved file:", output_path, "\n")
  cat("Original genes:", nrow(candidate_data), "\n")
  cat("After filtering:", nrow(filtered_data), "\n")
  
  filtered_data
}

# Step 3: run the filter for short/inter/long independently
results <- Map(
  filter_by_variability,
  input_path  = candidate_files,
  output_path = output_files,
  MoreArgs    = list(var_quantile = var_quantile, acf_quantile = acf_quantile)
)

# Step 4: final checks
lapply(results, head)
lapply(results, dim)
