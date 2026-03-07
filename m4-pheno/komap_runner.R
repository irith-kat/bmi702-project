#!/usr/bin/env Rscript
# komap_runner.R — called by komap_skill.py via subprocess
# Usage: Rscript komap_runner.R '<json_payload>'
#
# Required payload fields: ehr_path, main_surrogates, output_dir,
#   target_code, nm_disease, nm_utl, nm_corrupt_code
# Optional: target_cui, nm_corrupt_cui, codify_feature, nlp_feature

suppressPackageStartupMessages({
  library(KOMAP)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(mclust)
})

payload <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = FALSE)

ehr_data <- read.csv(payload$ehr_path, stringsAsFactors = FALSE)
out_dir  <- payload$output_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Add utl (healthcare utilisation) as a concept code ---------------------
# KOMAP requires a 'utl' column in the covariance matrix. It is computed as
# total distinct visit days per patient and must be present in ehr_data as a
# concept_code row before gen_cov_input aggregates the data.
utl_rows <- ehr_data %>%
  dplyr::distinct(patient_num, days_since_admission) %>%
  dplyr::mutate(concept_code = "utl")
ehr_data <- rbind(ehr_data, utl_rows)

# --- gen_cov_input -----------------------------------------------------------
# rollup_dict = NULL: codes are already rolled up by Python (via mapping.py).
#   gen_cov_input handles NULL by using each code as its own group.
# filter_df must be an empty data frame (not NULL): gen_cov_input calls
#   colnames(filter_df) unconditionally before any null-check.
input_cov <- gen_cov_input(
  ehr_data,
  rollup_dict     = NULL,
  filter_df       = data.frame(code = character(0), freq = integer(0)),
  main_surrogates = payload$main_surrogates,
  train_ratio     = 0.5
)

train_cov_path <- file.path(out_dir, "train_cov.csv")
valid_cov_path <- file.path(out_dir, "valid_cov.csv")
write.csv(input_cov$train_cov, train_cov_path)
write.csv(input_cov$valid_cov, valid_cov_path)
result <- list(train_cov_path = train_cov_path, valid_cov_path = valid_cov_path)

# --- Build dat_part (wide log-count matrix) from ehr_data -------------------
dat_part <- ehr_data %>%
  dplyr::count(patient_num, concept_code) %>%
  mutate(log_count = log1p(n)) %>%
  select(-n) %>%
  pivot_wider(names_from = concept_code, values_from = log_count, values_fill = 0)

# --- KOMAP_corrupt -----------------------------------------------------------
codify_feature <- if (!is.null(payload$codify_feature)) unlist(payload$codify_feature) else NULL
nlp_feature    <- if (!is.null(payload$nlp_feature))    unlist(payload$nlp_feature)    else NULL

out <- KOMAP_corrupt(
  input_cov$train_cov,
  input_cov$valid_cov,
  is.wide         = TRUE,
  target.code     = payload$target_code,
  target.cui      = payload$target_cui,
  nm.disease      = payload$nm_disease,
  nm.utl          = "utl",
  nm.multi        = NULL,
  nm.corrupt.code = payload$nm_corrupt_code,
  nm.corrupt.cui  = payload$nm_corrupt_cui,
  dict            = NULL,
  codify.feature  = codify_feature,
  nlp.feature     = nlp_feature,
  pred            = TRUE,
  eval.real       = FALSE,
  eval.sim        = FALSE,
  dat.part        = dat_part,
  nm.id           = "patient_num"
)

if (!is.null(out$est$long_df)) {
  p <- file.path(out_dir, "coefficients.csv")
  write.csv(out$est$long_df, p, row.names = FALSE)
  result$coefficients_path <- p
}

if (!is.null(out$pred_prob)) {
  for (nm in c("pred.score", "pred.prob", "pred.cluster")) {
    key <- gsub("\\.", "_", nm)
    p   <- file.path(out_dir, paste0(key, ".csv"))
    write.csv(out$pred_prob[[nm]], p, row.names = FALSE)
    result[[paste0(key, "_path")]] <- p
  }
}

cat(toJSON(result, auto_unbox = TRUE, null = "null"))
