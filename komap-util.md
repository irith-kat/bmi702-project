# KOMAP Utility

1. **Feature Discovery with ONCE:** Define the target codified features and/or narrative features. This step is done via the [ONCE Web App](https://shiny.parse-health.org/ONCE/)
   - Narrative features: standardized clinical concepts represented by concept unique identifiers (CUIs) according to the UMLS
   - Codified features: Phecodes mapped from ICD codes, grouped CPT procedure codes, etc.
   - Input:
     - Disease of interest
   - Output:
     - Codified feature list
     - Narrative feature list (NLP)
   - Steps to use ONCE web app:
      1. Enter disease of interest in search box
      2. CUIs and Phecodes will automatically populate in search box (this main CUI will determine the NLP features, and this main Phecode will determine the codified features)
      3. Search to generate feature lists:
         1. NLP Features
         2. Codified Features
      4. Download feature lists
         - To use ONCE-selected features in phenotyping, use features where the `phenotyping_features` column value is `true`; the threshold used to select these features is the most stringent, so only the most relevant potential features are selected to avoid noise in the phenotyping algorithm
2. **Generate covariance inputs:** Summarize raw EHR data into the privacy-preserving matrices required for the KOMAP algorithm
  - Input:
      - Raw longitudinal EHR data
      - A rollup dictionary (ICD to PheCode)
      - Some frequency filters
      - The main surrogate code <- from ONCE
  - Output:
      - `train_cov` and `valid_cov`
    ```
    data(ehr_data)
    data(rollup_dict)
    data(filter_df)
    input_cov <- gen_cov_input(ehr_data, rollup_dict, filter_df, main_surrogates = 'PheCode:250', train_ratio = 1/2)
    input_cov$train_cov, input_cov$vaid_cov
    ```
3. **Model training and score prediction:** Estimate regression coefficients and predict disease probabilities for each inputted patient
   - Input:
     -  Covariance matrices `train_cov` and `val_cov` from previous step
     - Wide-format log-counts for the patients you want to score (`dat.part`)
   - Output:
     - Regression coefficients
     - Predicted disease scores
     - Predicted disease probabilities
     - Predicted disease labels (using gaussian mixture model)
    ```    
    codify.feature <- codify_RA$Variable[codify_RA$high_confidence_level == 1]
    nlp.feature <- cui_RA$cui[cui_RA$high_confidence_level == 1]
    input.cov.train <- cov_RA_train_long
    input.cov.valid <- cov_RA_valid_long
    
    target.code <- 'PheCode:714.1'
    target.cui <- 'C0003873'
    nm.corrupt.code <- 'corrupt_mainICD'
    nm.corrupt.cui <- 'corrupt_mainNLP'
    nm.utl <- 'utl'
    nm.pi <- 'pi'
    nm.id <- 'patient_num'
    nm.y <- 'Y'
    dat.part <- dat_part
    
    out <- KOMAP_corrupt(input.cov.train, input.cov.valid, is.wide = TRUE, target.code, target.cui, 
                       nm.disease = 'RA', nm.utl, nm.multi = NULL, nm.corrupt.code = nm.corrupt.code, 
                       nm.corrupt.cui = nm.corrupt.cui, dict_RA, 
                       codify.feature = codify.feature, nlp.feature = nlp.feature,
                       pred = TRUE, eval.real = FALSE, eval.sim = FALSE,
                       dat.part = dat.part, nm.id = nm.id)
    ```
