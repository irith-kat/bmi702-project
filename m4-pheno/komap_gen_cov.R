# Parse payload
args <- commandArgs(trailingOnly = TRUE)
payload <- fromJSON(args[1])

ehr_data <- read.csv(payload$ehr_path)

# Optional rollup_dict
rollup_dict <- NULL
if (!is.null(payload$rollup_path)) {
  rollup_dict <- fromJSON(payload$rollup_path)
}

# Optional filter_df
filter_df <- NULL
if (!is.null(payload$filter_path)) {
  filter_df <- read.csv(payload$filter_path)
}

# Run KOMAP function
input_cov <- gen_cov_input(
  ehr_data = ehr_data,
  rollup_dict = rollup_dict,
  filter_df = filter_df,
  main_surrogates = payload$main_surrogates,
  train_ratio = payload$train_ratio
)

# Extract outputs
train_cov_path <- file.path(tempdir(), "train_cov.csv")
valid_cov_path <- file.path(tempdir(), "valid_cov.csv")

write.csv(input_cov$train_cov, train_cov_path, row.names = FALSE)
write.csv(input_cov$valid_cov, valid_cov_path, row.names = FALSE)

cat(toJSON(list(
  train_cov_path = train_cov_path,
  valid_cov_path = valid_cov_path
), auto_unbox = TRUE))