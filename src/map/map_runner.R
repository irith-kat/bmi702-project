#!/usr/bin/env Rscript
# Usage: Rscript map_runner.R <mat_csv> <note_csv> <output_csv> <main_icd_col>
#   mat_csv      — patient × feature count matrix (first column = patient ID)
#   note_csv     — patient × note count (first column = patient ID)
#   output_csv   — results destination (patient_id, score, phenotype)
#   main_icd_col — column in mat_csv to use as MAP's primary ICD surrogate

suppressPackageStartupMessages({
  library(MAP)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Usage: Rscript map_runner.R <mat_csv> <note_csv> <output_csv> <main_icd_col>")

mat_path     <- args[1]
note_path    <- args[2]
out_path     <- args[3]
main_icd_col <- args[4]

mat_df  <- read.csv(mat_path,  row.names = 1, check.names = FALSE)
note_df <- read.csv(note_path, row.names = 1, check.names = FALSE)

# MAP requires the primary ICD surrogate to be named exactly "ICD"
if (!(main_icd_col %in% colnames(mat_df))) {
  stop(paste("main_icd_col not found in mat columns:", main_icd_col))
}
colnames(mat_df)[colnames(mat_df) == main_icd_col] <- "ICD"

# make.names() converts to valid R identifiers ("PheCode:714.1" → "PheCode.714.1",
# "250.2" → "X250.2") so MAP's internal formula builder doesn't error
other_cols <- colnames(mat_df) != "ICD"
colnames(mat_df)[other_cols] <- make.names(colnames(mat_df)[other_cols])

# Restrict to patients present in both inputs
common_ids <- intersect(rownames(mat_df), rownames(note_df))
mat_df  <- mat_df[common_ids, , drop = FALSE]
note_df <- note_df[common_ids, , drop = FALSE]

# MAP expects sparse matrices
mat  <- Matrix(as.matrix(mat_df),  sparse = TRUE)
note <- Matrix(as.matrix(note_df), sparse = TRUE)

# Fit Poisson/normal mixture models; res$cut.MAP is the prevalence-based threshold
res <- MAP(mat = mat, note = note, verbose = FALSE)

out <- data.frame(
  patient_id = common_ids,
  score      = as.numeric(res$scores),
  phenotype  = as.integer(res$scores >= res$cut.MAP)  # 1 = above prevalence cutoff
)

write.csv(out, out_path, row.names = FALSE)
