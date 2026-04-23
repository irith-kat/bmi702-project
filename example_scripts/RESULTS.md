# Results and takeaways from HF test run v1

The aim of this run was to validate that all the components of the pipeline were adequately chained together, and explore the different failure modes of the pipeline (compute, time, data retrieval, NLP processing, model training...) to find the best options to bake into the skills for future runs.

Quite a bit of time was spent on hyperparameter tuning for LATTE, which is not straightforward and depends a lot on the signal present on the dataset, the silver labels used, the amount and distribution of gold labels.

A recurring setting was chosen for this pipeline validation as it is the most complex, so it allows us to test the full pipeline and all of its components, and understand Mode 2 for LATTE well.

## Takeaways
### Pipeline
- The code can be organized in a pipeline that runs end to end, diverse failure modes were identified and corrected in the skills and sample code so that future runs have an easier time.
- LATTE can extract some signal from data, but it is very much dependant on the study design and the chosen variables. MAP is more straightforward. This will require the agent to work longer on optimizing LATTE on future tasks, both hyperparameters and the chosen variables.
- Silver label coverage needs to be assessed before commiting to training.
- Mode 2 needs more controls, this can be achieved by evaluating the content of the gold labels and sampling more if they are mostly positive.
- There is training instability in LATTE, which requires choosing and tuning the proper weights. I indicated the agent to run up to 15 iterations on a single split,tuning parameters and inputs, before going over to CV.

### LATTE-specific
There is stochastic gradient collapes in the pipeline, some of the folds get stuck. This yields an average AUC of 0.61, while the non-collapsed average is rather closer than 0.67. The issue in this particular scenario is that we do not have a lot of control cases for LATTE to learn, most of the people have the recurrent event. This could be that MIMIC is indeed quite severe or that GEMINI is overlabeling. The little amount of negatives makes AUC very variable.

### LATTE Configuration (validated via 10-run tuning experiment)
**Baseline date:** 2100-01-01 (study-wide anchor; MIMIC-IV dates are shifted to ~2100–2200)
**Month window:** 3 (LATTE paper default)
**Feature codes:** ONCE feature_codes (64 codified + 115 NLP CUIs)
**Key codes:** LOINC:33762-6, ShortName:BNP (silver label proxy for decompensation)
**Epochs:** 35
**Epoch silver:** 8
**Embedding dim:** 50
**Layers incident:** "80" (single GRU layer — critical for small label sets; see tuning notes)
**weight_unlabel:** 0.015 (scaled to row ratio ≈ 1/72; default 0.2 causes gradient collapse)
**weight_prevalence:** 0.2
**weight_contrastive:** 0.1
**weight_smooth:** 0.1
**weight_additional:** 0.1
**flag_train_augment:** 1
**max_visits:** 25
**min_nonzero:** 20 (MAP feature matrix; set to 5 causes flexmix NaN log-likelihood)

### Hyperparameter Tuning Notes
A 10-run experiment was conducted to tune LATTE for this cohort. Key findings:
1. **Gradient collapse** (Run 1, AUC=0.500): Default `weight_unlabel=0.2` causes 14.4×
   unlabeled gradient dominance (72:1 row ratio with month_window=3). Fix: scale
   `weight_unlabel ≈ n_labeled / n_unlabeled ≈ 0.014`.
2. **LATTE checkpoint bug**: `a_semi_model_final.py` originally saved only the last 2
   epoch checkpoints. Patched to save from `epoch_silver` onward for true best-epoch
   selection. Worth +0.035 AUC.
3. **Single GRU layer** (Run 9, AUC=0.697): Switching from `layers_incident="80,80"` to
   `"80"` gained +0.043 AUC. With only ~30 positive cases the 2-layer GRU overfits.
4. **weight_unlabel and layers interact**: single-layer needs lower weight (0.015);
   dual-layer tolerates higher weight (0.025). Tune jointly.
5. **EPOCHS=35, EPOCH_SILVER=8 optimal**: 50 epochs overfits (best checkpoint at ~43);
   10 silver epochs reduces joint training time with no benefit.

Full tuning results: `logs/SUMMARY.md`
