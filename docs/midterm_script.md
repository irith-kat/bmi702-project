# Midterm Presentation Script (6 minutes)
## Rubric Mapping
| Rubric Criterion | Slide | Weight |
|---|---|---|
| Problem Statement & Background | 2–3 | 20 pts |
| Methodology | 4 | 20 pts |
| Baseline Results | 5 | 20 pts |
| Challenges and Solutions | 6 | 15 pts |
| Singular Ask for Feedback | 7 | 10 pts |
| Presentation Delivery | all | 15 pts |


## Script short (for presentation)
### Slide 1 — Title (10 sec)

### Slide 2 — Problem Statement (50 sec + 10 bufffer)
EHR data is the richest source of real-world clinical evidence we have — but it's messy. When building cohorts of interest, the most common approach is to filter by diagnostic codes or a combination of other codes, which is known to be unreliable due to billing biases and other factors. One study found that only 58% of patients with ICD codes for rheumatoid arthritis were confirmed cases on manual chart review.

Going from raw EHR tables to a validated cohort requires many steps like code vocabulary standardization, NLP on unstructured notes, and probabilistic phenotyping algorithms that are complex R packages most researchers run manually, if at all.

Our project aims to build a well-tooled agent that can handle and streamline this tedious process, opening clinical data research to less computational people.

### Slide 3 — Background & Related Work (30-40 sec)
Since the advent of LLMs and AI agents, the way we interact with data has been fundamentally changed. The barrier of natural-language to SQL has largely been solved: LLM agents can write queries, read results, and validate them in a loop, as shown by the M3 and M4 project by MIT Critical Data. Tools and skills have turned LLMs into truly AI Scientists, with ToolUniverse by Zitnik lab or AutoResearchClaw by Aiming Lab becoming popular. We identified a gap in the existing projects regarding proper agentic cohort building in real world data, which we aim to fill with this project. We envision it as a sub-agent or a module that could be integrated with wider end-to-end agentic clinical research pipelines, augmenting their capabilities.

### Slide 4 — Methodology (90 sec)
To build this project, we use MIMIC as a proof of concept dataset, although it is prepared to work with any SQL based clinical database. We leverage diverse state of the art projects:
- M4: M4 provides secure natural-language SQL access to MIMIC-IV (and any SQL-based clinical database).
- MAP, LATTE: algorithms built by CELEHS lab here in Harvard. MAP scores patients probabilistically using co-occurrence of structured codes and NLP signal, and LATTE performs incident phenotyping. They are R packages, so we make them accessible from Python.
- ONCE: for clinical feature selection, we use ONCE, a tool also from CELEHS that retreives and ranks related terms to a condition of interest.
- Preprocessing: rollup functions that normalize raw codes, currently ICD to PheCode, CPT to CCS, RxNorm to ingredient. The key design choice here is that the LLM handles dataset-specific schema (which column is the patient ID, which is the date) while our tools handle vocabulary translation. These tools will be expanded, and a skill included to teach the agent how to build new ones for custom vocabs.
- NLP: there si a lot of signal in the unstructured notes. To leverage them in an efficient way we process them with MedSpaCy, which allows us to extract condition-relevant mentions. We filter negated, uncertain, and family-history mentions. Although this pipeline is not as accurate as a custom-trained LLM for clinical NLP, it is way more computationally efficient (runs on a laptop in a few hours for a decent subset of MIMIC) and the signal picked up is sufficient for proof of concept.

Everything is orchestrated via Claude Code skills: a session orchestrator that defines a reproducible structure for script generation, plots, bias mitigation... as well as an interview to the researcher before starting; the preprocessing tools, note NLP, and MAP execution. From a single research prompt, the agent builds a reproducible protocol and runs the full pipeline.

### Slide 5 — Baseline Results (50 sec)

Script:
We ran the full pipeline on a few diseases including Multiple Sclerosis and Reumathoid Arthritis with successful results. MAP produces a bimodal score distribution (show a plot) which is the expected signature of a well-separated phenotype and confirms the algorithm is finding signal. Preliminary comparisons of the RA cohort on MIMIC with the example the MAP authors provided show matching results, establishing external validity.

Our baseline comparison is against what M4 alone gives you: a naive ICD code filter, with difficulty identifying other modality codes as it has no tools to access vocabulary definitions or standardize them. <add some numbers if we have them on how the vanilla M4 would work, or simply show the reduction of cases from simple code filtering to after MAP, to motivate the pipeline.>

We are in the process of accurately quantifying the importance of adding unstructured notes, but for now we can see how when NLP is included, the distribution sharpens and patients are classified more confidently in both directions.


### Slide 6 — Challenges & Solutions (45 sec)

Script:
For the second half, our focus is integrating LATTE for temporal phenotyping and deepening the evaluation framework.

The hardest open challenge is evaluation: how do we evaluate the pipeline as a whole and the value it adds? we are not developing a new phenotyping algorithm, but rather making it autonomous and accessible. More on this in the next slide.

A second challenge will be temporality and data leakage, that will come with our next step of integrating LATTE. Ensuring the agent is not grabbing information from the future can be problematic, and we will need to explore which are the right guardrails beyond prompt engineering to ensure proper cohort temporal quality.

One final challenge is context rot, which is a common issue in multistep agentic pipelines that think and reason. We are seeing that in our simple MAP phenotyping examples, we already reach between 50-75% of Claude Code context, so we will need to explore if the native /compact summarization method is enough or we need to offload individual tasks to subagents to keep the main context as slim as possible.

### Slide 7 — Singular Ask for Feedback (30 sec)

Script:
Our specific ask for feedback: **When a system chains together validated algorithms rather than building new ones, what is a proper evaluation method?**

There is no ground truth for cohort quality in the traditional sense. We have had 3 ideas:
1. Algorithm Fidelity: does the agent invoke MAP the same way a manual run would? How do we compare objectively?
2. Ablation: does each layer of the pipeline (preprocessing, NLP, phenotyping algorithms...) add measurable signal?
3. External validity: do our results on a target condition match the sensitivity and specificity reported in the original MAP paper?

We have preliminary done all three of these, but we're uncertain whether that is enough, or whether there is an expected standard for this type of system evaluation in clinical informatics. Any frameworks or precedents from your work would be very helpful."




## Script long (processed my me, but did not want to delete as we might be able to use some parts for the report)

### Slide 1 — Title (10 sec)
**[Title slide]**
> "Autonomous Phenotyping for Clinical Research Workflows"

### Slide 2 — Problem Statement (60 sec)
Script:
EHR data is the richest source of real-world clinical evidence we have — but it's notoriously messy. The most common approach to building a patient cohort is to filter by diagnostic codes or a combination of other codes, and that is known to be unreliable due to billing biases and other factors. One study found that only 58% of patients with two or more ICD codes for rheumatoid arthritis were confirmed cases on manual chart review.

The barrier of natural-language SQL has largely been solved — LLM agents can write queries, read results, and validate them in a loop. But a full clinical pipeline is a different challenge. Going from raw EHR tables to a validated cohort requires code vocabulary standardization, NLP on unstructured notes, and probabilistic phenotyping algorithms that are complex R packages most researchers run manually, if at all.

We argue this is exactly the kind of multi-step, expert-knowledge-heavy workflow that a well-tooled agent should handle and streamline. Imagine being able to phenotype 100s of diseases in your EHR, or generate robust cohorts on demand, and to be able to differentiate billing coding from real disease incidence, in an autonomous way and without computational expertise. That is the problem we are tackling with this project.

### Slide 3 — Background & Related Work (30 sec)
We build on M4, which provides secure natural-language SQL access to MIMIC-IV (and any SQL-based clinical database). They did not focus on the cohort building part nor develop tools for note processing, which is the gap we aim to fill in this project. For phenotyping, we use MAP — Multimodal Automated Phenotyping — a weakly supervised Poisson mixture model that scores patients probabilistically using co-occurrence of structured codes and NLP signal. For feature selection, rather than letting the agent choose (which could be an option for fully automated workflows), we use ONCE — a clinical feature selection tool.

This project could be integrated as a sub-agent module to ToolUniverse or other clinical research agent to make it widely accessible after we validate the concept on MIMIC.

### Slide 4 — Methodology (90 sec)

Here's our architecture. At the bottom is M4 — structured query access. On top of that we built a preprocessing layer: rollup functions that normalize raw codes — currently ICD to PheCode, CPT to CCS, RxNorm to ingredient. The key design choice here is that the LLM handles dataset-specific schema — which column is the patient ID, which is the date — while our tools handle vocabulary translation. That separation makes the preprocessing layer potentially applicable beyond MIMIC.

For unstructured data, we use MedSpaCy guided by CUI targets from ONCE to extract condition-relevant mentions from discharge notes. We filter negated, uncertain, and family-history mentions. Although this pipeline is not as accurate as a custom-trained LLM for clinical NLP, it is way more computationally efficient and the signal picked up is sufficient for proof of concept. Future work could explore more advanced NLP approaches.

For phenotyping, we wrapped the MAP algorithm, R-native, in a subprocess bridge, and make it callable from Python. MAP takes all the features and produces a 0-to-1 probability score per patient.

Everything is orchestrated via Claude Code skills: a session orchestrator that defines a reproducible structure for script generation, plots, bias mitigation... as well as an interview to the researcher before starting; the preprocessing tools, note NLP, and MAP execution. From a single research prompt, the agent builds a reproducible protocol and runs the full pipeline."

### Slide 5 — Baseline Results (60 sec)

Script:
We ran the full pipeline on a few diseases including Multiple Sclerosis and Reumathoid Arthritis with successful results. MAP produces a bimodal score distribution (show a plot) which is the expected signature of a well-separated phenotype and confirms the algorithm is finding signal.

Our baseline comparison is against what M4 alone gives you: a naive ICD code filter, with difficulty identifying other modality codes as it has no tools to access vocabulary definitions or standardize them. <add some numbers if we have them on how the vanilla M4 would work, or simply show the reduction of cases from simple code filtering to after MAP, to motivate the pipeline.>

We are in the process of accurately quantifying the importance of adding unstructured notes, but for now we can see how when NLP is included, the distribution sharpens and patients are classified more confidently in both directions.

Preliminary comparisons of the RA cohort on MIMIC with the example the MAP authors provided show matching results, showing how phenotyping contributes to a more realistic cohort, that the agent is chaining queries and scripts correctly and establishing external validity.

### Slide 6 — Challenges & Solutions (45 sec)

Script:
For the second half, our focus is integrating LATTE for temporal phenotyping and deepening the evaluation framework.

The hardest open challenge is evaluation: how do we evaluate the pipeline as a whole and the value it adds? we are not developing a new phenotyping algorithm, but rather making it autonomous and accessible. More on this in the next slide.

A second challenge will be temporality and data leakage, that will come with our next step of integrating LATTE, the temporal phenotyping algorithm, which identifies when a patient first develops a condition. Ensuring the agent is not grabbing information from the future, especially if the database has been anonymized and dates shuffled a bit, can be problematic, and we will need to explore which are the right guardrails beyond prompt engineering to ensure proper cohort temporal quality.

One final challenge is context rot, which is a common issue in multistep agentic pipelines that think and reason. We are seeing that in our simple MAP phenotyping examples, we already reach between 50-75% of Claude Code context, so we will need to explore if the native /compact summarization method is enough or we need to offload individual tasks to subagents to keep the main context as slim as possible.

### Slide 7 — Singular Ask for Feedback (30 sec)

Script:
Our specific ask for feedback: **When a system chains together validated algorithms rather than building new ones, what is a proper evaluation method?**

There is no ground truth for cohort quality in the traditional sense. We have had 3 ideas:
1. Algorithm Fidelity: does the agent invoke MAP the same way a manual run would? (we can mention we have already done this, we started with a manual run). How do we compare objectively?
2. Ablation: does each layer of the pipeline add measurable signal over the layer below? (preliminary tests show yes, which is what motivates the building of this agentic pipeline).
3. External validity: do our results on a target condition match the sensitivity and specificity reported in the original MAP paper?

We have plans for all three of these, but we're uncertain whether that is enough, or whether there is an expected standard for this type of system evaluation in clinical informatics. Any frameworks or precedents from your work would be very helpful."


## Evaluation Strategy detailed (Claude, not refined)

The core challenge: **no ground truth, not building algorithms, just chaining them.** Three-layer answer:

### Layer 1 — Algorithm fidelity (does the agent invoke MAP correctly?)
Run MAP manually on the same data and compare output. If agent-invoked MAP produces the same score distribution as manual MAP, the orchestration is correct. This is a regression test — not "is the cohort right," but "did we call the algorithm right."

### Layer 2 — Ablation (does adding each layer help?)
Define a weak baseline (ICD ≥2 codes), then compare:
- Baseline: ICD filter only (what M4 alone gives you)
- Step 1: MAP on structured codes only
- Step 2: MAP + NLP from discharge notes
Compare cohort size, score distribution shape, and % of patients that appear in one tier but not the other. The bimodal score distribution spreading when NLP is added is itself evidence of added signal.

### Layer 3 — Reference standard (is the cohort clinically plausible?)
Use the MAP original paper as proxy ground truth. The MAP paper validated on specific conditions (e.g., RA, T2DM). If our agent-produced cohort on the same condition with the same data produces comparable AUC/sensitivity/specificity to what the original paper reports, that is external validity. Published eMERGE PheWAS labels are also available for comparison on select conditions.

### What we cannot claim
We are not doing novel algorithm development, so we should be precise: our contribution is the *operationalization* — reproducibility, accessibility, and the reduction from multi-day manual pipeline to a single agent prompt. The validation should reflect that framing.
