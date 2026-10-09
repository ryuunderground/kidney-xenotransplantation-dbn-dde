### Convert all CSV files in a folder to TSV

library(readr)

# Folder containing the CSV files (change to your path)
input_dir <- "~/Documents/github/kidney-xenotransplantation-dbn-dde/preprocessing/DBN-DDE_inputs/"

# Where the TSV files should go (set equal to input_dir to save alongside originals)
output_dir <- "~/Documents/github/kidney-xenotransplantation-dbn-dde/preprocessing/mev_input/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

csv_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)

for (f in csv_files) {
  df <- read_csv(f, show_col_types = FALSE)
  
  out_name <- sub("\\.csv$", ".tsv", basename(f))
  out_path <- file.path(output_dir, out_name)
  
  write_tsv(df, out_path)
  cat("Converted:", f, "->", out_path, "\n")
}
