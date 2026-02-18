# Brainstorming log
Different ideas we explored during the week, all based on building on top of M4 (https://github.com/hannesill/m4/tree/main), an MCP system that allows AI agents to interact with Physionet databases.

## Idea 1: Phenotyping MCP module for agentic clinical research
Build an MCP module that is capable of performing agentic phenotyping for clinical research, based on a combination of billing codes and NLP notes. It would either be a standalone module that can be integrated into M4, or an extension of the M4 project itself, and it would allow for much more accurate phenotyping than just using billing codes.

We are considering models like:
- LATTE (https://pubmed.ncbi.nlm.nih.gov/38264714/) - takes care of longitudinal progression of disease, temporal labeling
- KOMAP (https://github.com/xinxiong0238/KOMAP/tree/master, https://www.medrxiv.org/content/10.1101/2023.09.29.23296239v2.full) - Takes care of labelling the patient as a whole.
- ONCE (https://www.medrxiv.org/content/10.1101/2023.09.29.23296239v2.full) - Takes care of the feature selection, is part of the KOMAP pipeline.

The result would be an augmented end-to-end research pipeline that leverages all the power of M4 for interacting with SQL databases, and the agentic skills to perform autonomous research, with the added benefit of robust phenotyping and clinical notes interaction, which is at the moment missing in M4.

For evaluation, we could:
- Clinical Drift Detection Agent: An autonomous agent can detect clinical drift (where hospital coding behavior changes over time) by monitoring the delta between billing-based cohorts and LATTE-based temporal phenotypes.
- Study comparison, automated vs existing results on MIMIC.
- Automatic paper crawler and replicator on MIMIC.

## Idea 2: Multi-agent system for Target Trial Emulation
- Our initial idea, use M4 and give it the skills missing (access to clinical trials gov, statistic packages, guidelines, clinical expertise from PubMed...) to be able to do robust TTE.
- ClaudeCode or GeminiCLI with M4 attached can already do decent retrospective studies, with audit trails that are ongoing progress.
- Some ideas to use notes for validation of edge inclusion/exclusion cases, do cross-database validations, although LLM+M4 can already do the latter if prompted to do so.
- The main problem is that we discovered TrialGenie, an agentic system currently published as a pre-print but without code, the does exactly this. Do differentiate (even if we would need to build the code from scratch) we could add M4 to the system they envisioned, and integrate notes and human-in-the-loop checkpoints, with TrialGenie lacks.
- Also has the risk that in 1-2 years, when Claude 5.5 or GPT 6 are out and the generalist agents can orchestrate themselves, spawn specific sub-agents (which they already can), this project would be eaten by the generalist agents + MCPs to interact with data (like M4) + code execution capabilities + agent skills to ground the procedures. This made us realize that it is better to focus on the infrastructure creation, that gets better as the LLMs get better, than trying to "compete" with Anthropic or OpenAI with agentic orchestration.

## Idea 3: Autonomous research question generator
- Instead of starting with a question, let an agent find the questions that need answering from the raw data, by automatically crawling MIMIC/eICU and comparing observations against established clinical guidelines, previous research...
- Establish a solid research protocol to tackle them. E.g., “I found 400 cases where Antibiotics were delayed by 4 hours vs. 1 hour. Should I run a TTE to see if this impacted mortality?”
- Bottom-up hypothesis discovery (as opposed to literature-based)

## Idea 4: Clinical failures auditor
- Explaining the why behind a "Clinical failure", we would be moving away from population-level trial emulation and into individualized quality control. We could use Sepsis 1-hour antibiotic rules or other procedures in the ICU like DKA as a test case. 
- The idea would be to do post-mortem audits on the patients in the database, using M4 as the interaction, and allow the hospital to have in real time aggregates of why patients died, if protocols are being systematically violated, why... It would focus on ICU data, and play detective in there. The aggregation of results could allow targeted workshops to the hospital population, if it is due to a lack of personel, lack of material, procedural barriers... Also, sometimes clinical "failures" are justified, and this tool could detect the common justifications.

Risks:
- The notes are not complete enough to be able to infer the risks of the failures
- There are not enough failures documented


