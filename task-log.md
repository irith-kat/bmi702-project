Here we can note progress on all tasks that are open and being done:

## Online Narrative and Codified feature Search Engine (ONCE)
I could not find open source code for ONCE sadly, only the implemented shiny app: https://shiny.parse-health.org/ONCE/. Nevertheless, since it is the very first step of doing research, as early as defining the clinical question and disease of interest, it is reasonable to direct the researcher to use the app, and apply his judgement to select the most rellevant NLP words and CUIs. This aligns this pipeline with human-in-the-loop processes and allows the very basic definition of the disease to be grounded and agreed between human and agent.

The file `once.py` intakes the downloaded files and prepares them for analysis.

## NILE
I (Beni) applied for access to the JAVA package, as it open for research but not fully opensource. Once we get it, we will need to develop the wrapper that can take a set of raw notes and return the CUIs.

## Phenotyping pipeline definition
I found this repo that, even if from 2 years ago, it contains the whole pipeline to extract RWE from EHR explained, with some rellevant repos: https://github.com/celehs/Harnessing-electronic-health-records-for-real-world-evidence?tab=readme-ov-file. 

I decided that easy functions like pivoting datasets and things like this can be written by an agent on the fly based on some agent skills. In the m4-pheno package, we should only implement complex functions that either require calculations, R code, training things, mapping with external files, etc...

## Data preprocessing
EHR data is messy. Assuming all data is as clean as MIMIC limits the usability of the research agent. This section will list preprocessing tools useful in RWE generation pipelines from EHR.

### Roll up ICD10 codes to Phecodes
Implemented the `rollup_icd_to_phecode` function in `m4-pheno/mapping.py` to facilitate the transformation of raw EHR diagnosis codes (ICD-9/10) into clinical PheCodes. This function utilizes the `Phecode_map_v1_2_icd9_icd10cm.csv` reference to provide a high-fidelity mapping essential for downstream phenotyping models like KOMAP. 
