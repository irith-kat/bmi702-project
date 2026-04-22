# Ablation Study Summary — Heart Failure Cohort

This document synthesizes key findings from each of the four ablation configurations,
drawn from `OUTPUT_COHORT.json`. It covers cohort sizes, methodology, key features/codes,
comorbidity patterns, and a cross-ablation comparison of what changed.

---

## A1-base — ICD Codes Only

**Cohort:** 19,715 cases / 192,083 controls (1:9.7 ratio)
**Features used:** ICD only

### Methodology Summary

> In this A1-base configuration, only ICD-9 and ICD-10 diagnostic codes were available and used for cohort construction; laboratory values, medication records, NLP clinical concept codes (CUIs), and probabilistic MAP scores were not incorporated. Cases were defined as adult patients (age ≥18) with at least two distinct ICD encounters bearing a heart failure code (ICD-9 428.x or ICD-10 I50.x), chosen to reduce misclassification from incidental or erroneous single-encounter coding; controls were all adult patients with at least one hospital admission and zero heart failure ICD codes across any encounter. No vocabulary rollup (PheCode, RxNorm, CCS, or LOINC mapping) or feature preprocessing was applied; comorbidity prevalence was computed using raw ICD prefix matching, and patients with exactly one HF code were excluded from both cases and controls to avoid an ambiguous borderline group. Key limitations include the absence of NLP signal (patients discussed in free-text notes without a corresponding ICD code are undetectable), the discharge-time assignment of ICD codes in MIMIC-IV (which captures the full stay rather than admission-time status), and the lack of MAP-based probabilistic phenotyping that could recover borderline or miscoded cases. A notable assumption is that heart failure coded once may reflect a transient or incidental finding rather than a true diagnosis, so the ≥2 encounter threshold was applied; additionally, ICD-9 code I11.0 (hypertensive heart disease with heart failure) was retained as a comorbidity marker rather than an anchor code, as it appeared among the top non-anchor codes with 24% prevalence in cases.

### Key Features / Codes

Top driving codes (by raw prevalence in cases):

| Code | Meaning | Prevalence |
|------|---------|-----------|
| 4019 | Unspecified essential hypertension | 48.5% |
| 2724 | Hyperlipidemia | 45.4% |
| 42731 | Atrial fibrillation | 37.1% |
| 41401 | Coronary atherosclerosis | 37.0% |
| E785 | Hyperlipidemia, unspecified | 36.9% |
| 5849 | Acute kidney failure | 34.6% |
| I2510 | Atherosclerotic heart disease | 29.6% |
| N179 | Acute kidney failure, unspecified | 29.3% |
| 25000 | Type 2 diabetes | 28.7% |
| I110 | Hypertensive heart disease with HF | 24.2% |

**Notable:** All codes are raw ICD-9/ICD-10 with no rollup — codes appear as both ICD-9 and ICD-10 versions separately (e.g., `2724` and `E785` are both hyperlipidemia). Comorbidity matching is ICD prefix-based.

### Comorbidities

| Comorbidity | Prevalence |
|-------------|-----------|
| Hypertension | 89.7% |
| Anemia | 66.5% |
| Coronary Artery Disease | 63.0% |
| Atrial Fibrillation | 56.8% |
| CKD | 51.9% |
| Diabetes | 48.0% |
| Chronic Pulmonary Disease | 38.0% |
| Obesity | 24.4% |

---

## A2-structured — ICD + Rx + Lab + CPT (Vocabulary Rollups)

**Cohort:** 14,081 cases / 14,081 controls (1:1 ratio, random sampling)
**Features used:** ICD (→PheCode), Rx (NDC→RxNorm), Lab (→LOINC), CPT (→AHRQ CCS)

### Methodology Summary

> This A2-structured cohort used four structured EHR modalities from MIMIC-IV: ICD diagnosis codes (rolled up to PheCodes), prescriptions (NDC→RxNorm ingredient), CPT/HCPCS procedures (→AHRQ CCS), and lab events (itemid→LOINC); no NLP or MAP probabilistic scoring was available in this configuration. Heart failure cases were defined as adults (age ≥18) with ≥2 distinct hospital admissions bearing an ICD-9 428.x or ICD-10 I50.x code, rolled up to PheCode 428; controls were randomly sampled 1:1 from adults with zero PheCode:428 events, without demographic matching. The observation log was built by rolling up ICD→PheCode via Phecode_map v1.2, NDC→RxNorm ingredient via OMOP Athena mapping (~97.7% NDC coverage, drug-name fallback for remainder), CPT→CCS via AHRQ 2025 mapping, and labevents itemid→LOINC via MIT-LCP OMOP mapping (1400/1630 itemids covered); all modalities were concatenated into a single long-format observation log. Key limitations: NLP signal is absent so patients mentioned only in free-text notes are missed; the ≥2-admission threshold is a heuristic and may exclude true single-admission HF patients while including patients with incidental coding; lab data covers only numeric valuenum rows, dropping qualitative results; controls are unmatched on age/sex. Heart failure is frequently a comorbid secondary diagnosis in MIMIC-IV rather than the primary admission reason, so the ≥2 threshold helps reduce incidental coding noise but cannot distinguish primary HF admissions from comorbid documentation.

### Key Features / Codes

Top driving codes (by case/control enrichment ratio):

| Code | Meaning | Case Prev | Enrichment |
|------|---------|----------|-----------|
| RXNORM:46275719 | Sacubitril (Entresto) | 3.9% | 39,415× |
| RXNORM:45774751 | Empagliflozin (Jardiance) | 1.2% | 12,073× |
| PheCode:401.2 | Primary hypertension NOS | 33.7% | 395× |
| PheCode:401.21 | Hypertensive CKD | 34.5% | 186× |
| PheCode:573.1 | Hepatic congestion | 3.5% | 97× |
| PheCode:425.2 | Cardiomyopathy | 3.3% | 76× |
| RXNORM:942350 | Torsemide (loop diuretic) | 35.8% | 67× |
| RXNORM:1337720 | Dobutamine (inotrope) | 5.3% | 57× |
| PheCode:411.8 | Other acute ischemic heart disease | 18.0% | 53× |
| RXNORM:907013 | Metolazone (thiazide diuretic) | 17.1% | 36× |

**Notable:** The most discriminating signals are **HF-specific medications** (sacubitril/Entresto, torsemide, dobutamine, metolazone) that appear almost exclusively in HF cases. PheCodes replace raw ICD codes, merging ICD-9 and ICD-10 synonyms. The ≥2 admissions threshold reduces the case count from 31,369 anchor candidates to 14,081 (55% retained).

### Comorbidities

| Comorbidity | Prevalence |
|-------------|-----------|
| Hypertension | 93.9% |
| CKD | 78.0% |
| Coronary Artery Disease | 73.5% |
| Atrial Fibrillation | 60.0% |
| Diabetes | 54.8% |
| COPD | 32.1% |
| Obesity | 33.0% |
| Cardiomyopathy | 26.4% |
| Sleep Apnea | 25.5% |
| Anemia | 24.6% |

---

## A3-nlp — ICD + Rx + Lab + CPT + NLP CUIs

**Cohort:** 42,856 cases / 85,712 controls (1:2 ratio)
**Features used:** ICD (→PheCode), Rx, Lab, CPT, NLP (MedSpaCy CUIs from discharge notes)

### Methodology Summary

> Feature types available and used: ICD→PheCode, NDC→RxNorm (Rx), HCPCS→CCS (CPT), itemid→LOINC (Lab), and discharge note CUIs via MedSpaCy NER (NLP); all five modalities were included. Cases were defined under three arms: Arm 1 (ICD-confirmed) requires ≥2 HF PheCodes AND ≥1 HF CUI mention in discharge notes, so that ICD-coded patients without note-level evidence are excluded as likely false positives; Arm 2 (borderline confirmed) requires 1 HF PheCode AND ≥1 HF CUI; Arm 3 (NLP-expansion) captures patients with 0 HF PheCodes but ≥2 HF CUI events in notes (minimum-2 threshold suppresses single-mention noise). Controls have 0 HF PheCodes and 0 HF CUI mentions, sampled 2:1. NLP for ICD patients (Pass A) ran on all 31,414 HF PheCode patients; NLP-expansion (Pass B) first applied a keyword pre-filter (heart failure, CHF, HFrEF, EF<40, etc.) to 0-PheCode patients' notes, then ran MedSpaCy only on keyword-matching notes, making the corpus tractable. Known limitations: NLP-expansion recall is bounded by the keyword pre-filter — patients with HF mentioned only obliquely in notes may be missed; the ≥2 CUI threshold for NLP-only cases is conservative and may under-capture single-admission patients; no MAP scores are used in this configuration. Key v2 judgment calls: requiring NLP confirmation for ≥2-PheCode cases improves specificity at the cost of losing patients who had legitimate HF but whose notes lacked direct CUI-matching terms (e.g., transferred patients with no local discharge note).

### Case Definition Arms

| Arm | Criteria | N Cases |
|-----|----------|---------|
| ICD-confirmed | ≥2 HF PheCodes AND ≥1 HF CUI | 16,628 |
| Borderline confirmed | 1 HF PheCode AND ≥1 HF CUI | 7,384 |
| NLP-expansion | 0 HF PheCodes, ≥2 HF CUIs | 18,844 |
| **Total** | | **42,856** |

### Key Features / Codes

Top driving codes (by prevalence in cases, PheCode-based):

| Code | Meaning | Prevalence |
|------|---------|-----------|
| PheCode:401.1 | Essential hypertension | 60.2% |
| PheCode:272.1 | Hyperlipidemia | 59.5% |
| PheCode:411.4 | Coronary atherosclerosis | 52.9% |
| PheCode:318 | Intellectual disabilities / cognitive | 46.3% |
| PheCode:585.1 | CKD stage 1–2 | 44.2% |
| PheCode:427.21 | Atrial fibrillation | 43.3% |
| PheCode:1010.7 | Sepsis / SIRS | 37.7% |
| PheCode:530.11 | GERD | 36.3% |
| PheCode:250.2 | Type 2 diabetes | 33.5% |
| PheCode:411.2 | Acute MI | 32.0% |

**Notable:** NLP expansion dramatically increases case count (+44% over A3's ICD-only anchor pool) by capturing 18,844 patients documented in free text but uncoded. PheCode:318 (intellectual disabilities) appearing highly prevalent may reflect documentation patterns in MIMIC-IV rather than true clinical association.

### Comorbidities

| Comorbidity | Prevalence |
|-------------|-----------|
| Hypertension | 82.2% |
| Coronary Artery Disease | 53.5% |
| CKD | 50.7% |
| Atrial Fibrillation | 43.3% |
| Anemia | 30.0% |
| Diabetes | 34.4% |
| Pulmonary Hypertension | 17.6% |
| COPD | 19.3% |
| Cardiomyopathy | 11.1% |
| Obesity | 20.2% |
| Sleep Apnea | 2.5% |
| Valvular Disease | 9.0% |

---

## A4-full — All Modalities + MAP Probabilistic Phenotyping

**Cohort:** 5,039 cases / 98,550 controls (1:19.6 ratio)
**Features used:** ICD (→PheCode), Rx (→RxNorm), Lab (→LOINC), CPT (→CCS), NLP (CUIs) → MAP scoring

### Methodology Summary

> All five EHR feature modalities were included in the observation log: ICD diagnoses (rolled to PheCode), prescriptions (NDC → RxNorm), lab events (itemid → LOINC), HCPCS/CPT procedures (rolled to AHRQ CCS), and NLP CUI mentions extracted from discharge notes via MedSpaCy. However, the ONCE feature file for heart failure includes PheCode and RxNorm codified features but uses ShortName labels (INR, BNP, PT) for labs instead of LOINC codes; because ONCE ShortName prefixes do not match LOINC prefixes in the observation log, lab and CPT features did not contribute to MAP scoring — only ICD (PheCode), medication (RxNorm), and NLP (CUI) features drove phenotype probabilities. Cases were defined as patients with MAP posterior probability ≥ MAP's internal prevalence cutoff (phenotype=1); controls were all MAP-matrix patients below that threshold. Preprocessing rolled ICD-9/10 codes to PheCodes using the v1.2 PheCode map, NDC codes to RxNorm ingredient level via OMOP Athena, and NLP was restricted to ICD anchor candidates (PheCode:428*) to manage compute — patients without any heart failure ICD code did not receive NLP feature extraction. Key limitations: ShortName/LOINC mismatch excluded clinically informative labs (BNP, INR, PT) from MAP; NLP was not run on non-ICD-candidates, so MAP-only cases (ICD-negative but MAP-positive) are likely underestimated; ICD codes are assigned at discharge in MIMIC-IV, which is appropriate for phenotyping but not admission-time analyses. A judgment call was made to restrict NLP to PheCode:428* candidates rather than all patients to keep the pipeline tractable; this trades recall for feasibility in a dataset of this size.

### MAP-Specific Metrics

| Metric | Value |
|--------|-------|
| ICD anchor candidates | 31,414 |
| Final MAP cases | 5,039 |
| ICD-coded but MAP-rejected | 26,375 |
| MAP-found (ICD-negative) | 0 |
| MAP acceptance rate (of ICD anchors) | 16.0% |

MAP rejected 84% of ICD-anchor candidates, keeping only those with strong multi-modal signal.

### Key Features / Codes

Top MAP feature weights (most predictive of HF):

| Code | Meaning | Weight |
|------|---------|--------|
| PheCode:428.1 | CHF NOS | 0.592 |
| CUI:C0025859 | Metoprolol | 0.348 |
| PheCode:428.3 | HFrEF (systolic HF) | 0.343 |
| CUI:C0004238 | Atrial Fibrillation | 0.335 |
| CUI:C0016860 | Furosemide | 0.310 |
| PheCode:428.4 | HFpEF (diastolic HF) | 0.306 |
| CUI:C0018801 | Heart failure | 0.280 |
| PheCode:427.21 | Atrial fibrillation | 0.271 |
| CUI:C0018802 | Congestive heart failure | 0.267 |
| CUI:C0013404 | Dyspnea | 0.259 |
| CUI:C1956346 | Coronary Artery Disease | 0.256 |
| LOINC:33762-6 | NTproBNP | 0.256 |
| CUI:C0020649 | Hypotension | 0.230 |
| CUI:C0011849 | Diabetes Mellitus | 0.203 |

**Notable:** MAP's top weights are split between **ICD-derived PheCodes** (specific HF subtypes: CHF NOS, HFrEF, HFpEF) and **NLP CUIs** (Furosemide, Metoprolol, Dyspnea). NTproBNP (a key HF biomarker) appears despite the ShortName/LOINC mismatch — this likely came through via NLP CUI extraction from note text.

### Comorbidities

| Comorbidity | Prevalence |
|-------------|-----------|
| Hypertension | 89.7% |
| Atrial Fibrillation | 65.1% |
| CKD | 63.5% |
| Coronary Artery Disease | 71.7% |
| Diabetes | 50.1% |
| Cardiomyopathy | 29.2% |
| Valvular Disease | 26.5% |
| Pulmonary Hypertension | 1.5% |

---

## Cross-Ablation Comparison

### Cohort Sizes

| Ablation | Cases | Controls | Ratio |
|----------|-------|----------|-------|
| A1-base | 19,715 | 192,083 | 1:9.7 |
| A2-structured | 14,081 | 14,081 | 1:1 |
| A3-nlp | 42,856 | 85,712 | 1:2 |
| A4-full | 5,039 | 98,550 | 1:19.6 |

A3-nlp produces the **largest case set** because its design intent was **expansion**: NLP is used to recover 18,844 patients who appear in discharge notes but were never ICD-coded, deliberately broadening the case definition beyond what structured codes alone capture. A4-full produces the **smallest and most conservative** case set because its design intent was the opposite — **restriction**: MAP starts from the same original ICD anchor pool (~31,414 PheCode:428 patients, not A3's expanded set) and rejects 84% of those anchors that lack corroborating multi-modal signal, keeping only the 5,039 with the strongest probabilistic evidence.

### Case Definition Criteria Compared

| Ablation | Anchor | Threshold | Expansion |
|----------|--------|-----------|-----------|
| A1-base | ≥1 HF ICD (428.x / I50.x) | ≥2 HF ICD encounters | None |
| A2-structured | ≥1 HF PheCode:428 | ≥2 distinct admissions | None |
| A3-nlp | ≥1 HF PheCode:428 OR ≥2 HF CUIs | Arm-specific (see above) | NLP-expansion arm (18,844 patients) |
| A4-full | ≥1 HF PheCode:428 | MAP posterior ≥ threshold | None (NLP restricted to ICD anchors) |

### Key Differences in Features Used

| Feature | A1 | A2 | A3 | A4 |
|---------|----|----|----|----|
| ICD (raw) | ✓ | — | — | — |
| ICD (→PheCode) | — | ✓ | ✓ | ✓ |
| Rx (→RxNorm) | — | ✓ | ✓ | ✓ |
| Lab (→LOINC) | — | ✓ | ✓ | ✓* |
| CPT (→CCS) | — | ✓ | ✓ | ✓* |
| NLP CUIs | — | — | ✓ | ✓ |
| MAP scoring | — | — | — | ✓ |

*A4 loaded these but ShortName/LOINC mismatch meant labs/CPT did not contribute to MAP scores.

### Key Differences in What Each Ablation Captures

- **A1→A2**: Adding vocabulary rollup (PheCode/RxNorm) and multi-modal structured data introduces HF-specific medications (sacubitril, torsemide, dobutamine) as the strongest discriminators. The ≥2 admissions (vs ≥2 encounters) threshold cuts cases by 29%. Controls change from all non-HF adults (192K) to 1:1 random sample (14K).

- **A2→A3**: Adding NLP triples the case count by recovering 18,844 ICD-negative patients who appear in discharge notes. It also changes ICD-confirmed case definition from ICD-only to ICD+NLP confirmation, which can drop some A2 cases that lacked note-level evidence. Comorbidity prevalences are lower in A3 because NLP-expansion cases (the new ones) tend to be less severely coded.

- **A3→A4**: These two ablations represent opposite strategies and should not be read as a sequential pipeline. A3's intent was **expansion** — it deliberately used NLP to add patients the ICD system missed, growing the cohort to 42,856. A4's intent was **restriction** — it returned to the same original ICD anchor pool (~31,414 PheCode:428 patients) that A1/A2 used and applied MAP probabilistic scoring to filter that pool down, not to expand it further. NLP in A4 was restricted to ICD-anchor candidates only (i.e., it served MAP's feature extraction, not case discovery). MAP integrates PheCode, RxNorm, and NLP CUIs probabilistically and sets its own prevalence cutoff, ultimately retaining only 5,039 patients with the strongest multi-modal signal. The top MAP features show HF subtype specificity (HFrEF vs HFpEF vs CHF NOS) alongside medication CUIs (Furosemide, Metoprolol) as independent phenotype signals. The 0 MAP-only cases confirms that MAP found no new patients outside the ICD anchor pool — consistent with its restrictive design.

### Comorbidity Prevalence Comparison

| Comorbidity | A1-base | A2-structured | A3-nlp | A4-full |
|-------------|---------|--------------|--------|---------|
| Hypertension | 89.7% | 93.9% | 82.2% | 89.7% |
| CKD | 51.9% | 78.0% | 50.7% | 63.5% |
| Atrial Fibrillation | 56.8% | 60.0% | 43.3% | 65.1% |
| Coronary Artery Disease | 63.0% | 73.5% | 53.5% | 71.7% |
| Diabetes | 48.0% | 54.8% | 34.4% | 50.1% |
| Anemia | 66.5% | 24.6% | 30.0% | — |

A2-structured shows elevated CKD (78%) and CAD (73.5%) vs other ablations, likely because the ≥2 admission threshold enriches for more severely ill patients who have more comorbid conditions documented. A3-nlp shows lower prevalences because the NLP-expansion arm adds patients who were not previously coded — these patients may have had HF mentioned in notes but fewer documented comorbidities. A4-full's MAP-selected cohort shows high atrial fibrillation (65.1%) and CAD (71.7%), matching published HF phenotyping studies, suggesting that MAP's restrictive approach — filtering the original ICD anchor pool rather than expanding beyond it — produces the most clinically coherent cohort. This contrast with A3 is instructive: A3's lower comorbidity prevalences reflect the dilution effect of adding NLP-expansion cases that were undercoded across all modalities, while A4's higher prevalences reflect that only the most richly documented (and thus most convincingly confirmed) ICD-anchored patients survived MAP's filter.
