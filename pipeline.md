## Module A: Preprocessing for Harmonization
The aim of these functions and skills is to teach the agent how to take raw EHR data and transform it into a format that can be effectively used for RWE generation. This includes mapping raw codes to standardized vocabularies, extracting relevant features, and handling missing data.

### Required Functions
* **`rollup.rollup_icd_to_phecode(df, icd_column)`** ✅: Maps raw ICD-9/10 to PheCodes. Handles MIMIC-IV's dot-less format automatically. https://phewascatalog.org/phewas/#phe12, filename `Phecode_map_v1_2_icd9_icd10cm.csv`
* **`rollup.rollup_rxnorm_to_ingredient(df, rxnorm_column)`** ⚠️ stub: Maps RxNorm codes to their active ingredients. Placeholder — RxNav API or local DB integration pending.
* **`rollup.rollup_loinc_to_test(df)`** ❌ not yet implemented: Maps LOINC codes to their corresponding test-level concepts.
* **`rollup.rollup_cpt_to_ccs(df, cpt_column)`** ✅: Maps CPT/HCPCS codes to CCS categories for procedures using AHRQ alphanumeric range matching. https://hcup-us.ahrq.gov/toolssoftware/ccs_svcsproc/ccscpt_downloading.jsp

### Optional Functions for Code Understanding (MCP)
MCP for understanding codes? that simply provides code definitions to the LLM to ensure it understands the medical concepts behind the codes it's working with. This could be a simple function that takes a list of codes and returns their definitions, which the agent can then use to inform its reasoning when selecting features or interpreting results.

* **`vocab.get_code_definition(code, source)`** ✅: Takes a medical code (ICD10CM, LNC, RXNORM) and returns its human-readable definition via the UMLS Terminology Services API.
* **`vocabulary.get_phecode_mapping(icd_code)`** ❌ not yet implemented: Provides the corresponding PheCode(s) for a given ICD code, which can help the agent understand how raw diagnosis codes relate to the phenotyping features it will use in models like KOMAP.
* **`vocabulary.get_rxnorm_ingredient(rxnorm_code)`** ❌ not yet implemented: Returns the active ingredient(s) for a given RxNorm code, aiding the agent in understanding medication data.
* **`vocabulary.get_loinc_test(loinc_code)`** ❌ not yet implemented: Provides the test-level concept for a given LOINC code, which can help the agent interpret lab test data in the context of phenotyping.
* **`vocabulary.get_ccs_category(cpt_code)`** ❌ not yet implemented: Returns the CCS category for a given CPT/HCPCS code, which can help the agent understand procedural data in the context of phenotyping.
* **`vocabulary.get_cui_mapping(term)`** ❌ not yet implemented: Uses a local or API-based lookup to find UMLS CUIs for medical terms (e.g., "RA" → C0003873).

### Required Agent Skills
* **Skill: `preprocessing-strategy`** ✅: Instructions on how to preprocess and harmonize EHR data, which ideal requirements it should fullfill, and how to use the above functions to achieve that. It should ensure the output from this step is in a format that can be easily fed into the phenotyping models (e.g., a DataFrame with patient IDs and their associated PheCodes, ingredients, tests, and procedures). It should also include logic for handling missing data, ensuring that the agent can still proceed with phenotyping even if some features are not available for all patients.

---

## Module B: Build Datamart
The aim of this module is to take the harmonized data and construct a broad, sensitivity-first feature matrix anchored on ONCE-curated features. ONCE output files (https://shiny.parse-health.org/ONCE/) are the required entry point for feature selection — ensuring it is grounded in validated clinical knowledge before any modeling occurs.

> NOTE: I could not find open source code for ONCE sadly, only the implemented shiny app: https://shiny.parse-health.org/ONCE/. Because of that, it will be kept as a manual step and asked to the researcher during the interview. This is because it is crucial to defining the clinical question and disease of interest, so it is reasonable to direct the researcher to use the app, and apply his judgement to select the most rellevant NLP words and CUIs. This aligns this pipeline with human-in-the-loop processes and allows the very basic definition of the disease to be grounded and agreed between human and agent.

### Required Functions

* **`once.get_once_features(codified_file, narrative_file)`** ✅: Parses ONCE output files (auto-detecting separator). Returns a dict with `codified` (full DataFrame), `narrative` (full DataFrame), `codified_list` (high-confidence Variable strings), `nlp_list` (CUI strings), and `nlp_target_cuis` (term+CUI dicts for MedSpaCy).

* **`once.parse_once_by_modality(once_features)`** ✅: Splits `codified_list` by vocabulary prefix into `phecode`, `rxnorm`, `loinc`, `ccs`, `shortname`, and `other` buckets (prefix stripped). Used to fan out ONCE features to each modality's rollup function.

* **`note_ner.extract_cui_features(notes_df, text_column, id_column, target_cuis)`** ✅: MedSpaCy NER pipeline. Takes clinical notes, extracts CUIs matching the ONCE narrative list, filters negated/uncertain/family-history mentions, and returns a long-format CUI feature table.

* **`note_ner.aggregate_features(cui_df, id_column, feature_column)`** ✅: Pivots the long-format CUI table into a wide patient×CUI count matrix suitable for joining onto `mat_df`.

* **`preprocessing.build_map_feature_matrix(phecode_df, once_phecodes, main_phecode, min_nonzero)`** ✅: Builds the MAP feature matrix from PheCode-rolled EHR data and ONCE feature list. Pivots to wide format, restricts to ONCE features, filters zero-count patients, and drops sparse features (while always retaining the anchor PheCode).

* **`preprocessing.build_note_proxy(admissions_df, study_index)`** ✅: Builds a `note_count` proxy from admission counts when MIMIC-IV-Note is unavailable. Fills missing patients with 1 to satisfy MAP's non-zero Poisson denominator requirement.

* **`preprocessing.prepare_map_inputs(ehr_df, notes_df, icd_col, include_nlp, ...)`** ✅ legacy wrapper: End-to-end function combining rollup, optional NLP, and note counting into `mat_df` + `note_df`. Prefer calling `build_map_feature_matrix` and `build_note_proxy` directly for more control.

### Required Agent Skills
* **Skill: `build-datamart`** ✅: Instructions for the agent to prioritize sensitivity over specificity in this initial query, ensuring that the true cohort is fully captured even if it includes many false positives. It should include available phecodes, ingredients, tests, and procedures related to the disease of interest (provided by user from ONCE or other sources) to create a broad "Data Mart" of potential patients. The agent should also be instructed to optionally include NLP features (CUIs) if available, although it might be better to first identify the cohort based on codified data and then refine it with NLP features in a second step, depending on the computational resources and dataset size.

---

## Module C: Phenotyping Strategy
The aim of this module is to take the Data Mart produced by Module B and apply a phenotyping algorithm to identify the true disease cohort. The agent selects between MAP (semi-supervised, requires labeled anchor) and nothing based on data availability and task requirements. LATTE might be added in a future step.

### Required Functions

* **`map.run_map(mat_df, note_df, main_icd_col)`** ✅: Python→R bridge for the MAP (Multimodal Automated Phenotyping) algorithm. Writes CSVs to a temp dir, calls `map_runner.R` via subprocess, returns a DataFrame with `patient_id`, `score`, and `phenotype` columns.

### Required Agent Skills
* **Skill: `map-phenotyping`** ✅: Instructions for running the MAP algorithm on a prepared `mat_df` + `note_df`. Covers when to use MAP vs KOMAP, how to interpret the output scores, and how to choose a probability threshold for case/control labeling.

* **Skill: `phenotyping-strategy`** ❌ not yet implemented: Logic to decide between rule-based, semi-supervised (MAP) approaches based on the availability of labels and the task at hand.

EXTRA STEPS IN PROGRESS...
