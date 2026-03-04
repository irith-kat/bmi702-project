## Module 1: Preprocessing for Harmonization

### Required Functions
* **`mapping.rollup_icd_to_phecode(df, type)`**: Maps raw ICD-9/10 to PheCodes
* **`mapping.rollup_rxnorm_to_ingredient(df)`**: Maps RxNorm codes to their active ingredients.
* **`mapping.rollup_loinc_to_test(df)`**: Maps LOINC codes to their corresponding test-level concepts.
* **`mapping.rollup_cpt_to_ccs(df)`**: Maps CPT/HCPCS codes to CCS categories for procedures. It accepts a df, but might need to be adapted to work with summary level data if needed.

## Optional functions for code understanding (MCP)
MCP for understanding codes? that simply provides code definitions to the LLM to ensure it understands the medical concepts behind the codes it's working with. This could be a simple function that takes a list of codes and returns their definitions, which the agent can then use to inform its reasoning when selecting features or interpreting results.

* **`vocabulary.get_code_definition(code, vocab)`**: Takes a medical code (ICD, LOINC, RxNorm) and returns its human-readable definition to help the agent understand the medical concepts it is working with. This can be implemented using a local mapping file or an API call to a medical vocabulary service.
* **`vocabulary.get_phecode_mapping(icd_code)`**: Provides the corresponding PheCode(s) for a given ICD code, which can help the agent understand how raw diagnosis codes relate to the phenotyping features it will use in models like KOMAP.
* **`vocabulary.get_rxnorm_ingredient(rxnorm_code)`**: Returns the active ingredient(s) for a given RxNorm code, aiding the agent in understanding medication data.
* **`vocabulary.get_loinc_test(loinc_code)`**: Provides the test-level concept for a given LOINC code, which can help the agent interpret lab test data in the context of phenotyping.
* **`vocabulary.get_ccs_category(cpt_code)`**: Returns the CCS category for a given CPT/HCPCS code, which can help the agent understand procedural data in the context of phenotyping.
* **`vocabulary.get_cui_mapping(term)`**: Uses a local or API-based lookup to find UMLS CUIs for medical terms (e.g., "RA" $\rightarrow$ C0003873).

### Required Agent Skills
TBD

---

## Module 2: Cohort Construction

### Required Functions

* **`nile.extract_cuis(note, ....)`**: Wrapper for Narrative Information Linear Extraction to pull CUIs from text. Arguments to be defined based on the NILE implementation and the needs of the phenotyping models. This function can be run on the full notes or just a subset based on the datamart.

* **`komap.run_komap(args TBD)`**: The Python-R bridge that trains the KOMAP model on the Data Mart to identify the true Disease Cohort.

TBD

### Required Agent Skills

* **Skill: `build_datamart`**: Instructions for the agent to prioritize sensitivity over specificity in this initial query, ensuring that the true cohort is fully captured even if it includes many false positives. It should include available phecodes, ingredients, tests, and procedures related to the disease of interest (provided by user from ONCE or other sources) to create a broad "Data Mart" of potential patients. The agent should also be instructed to optionally include NLP features (CUIs) if available, although it might be better to first identify the cohort based on codified data and then refine it with NLP features in a second step, depending on the computational resources and dataset size.

* **Skill: `phenotyping-strategy`**: Logic to decide between rule-based, weakly supervised (KOMAP), or semi-supervised approaches based on the availability of labels.
* **Skill: `temporal-logic`**: Instructions on setting "Time-Zero" at treatment initiation and ensuring the indication conditions exist prior to that date .

EXTRA STEPS IN PROGRESS...