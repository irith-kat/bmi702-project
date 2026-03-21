# Phenotyping pipeline definition
I found this repo that, even if from 2 years ago, it contains the whole pipeline to extract RWE from EHR explained, with some rellevant repos: https://github.com/celehs/Harnessing-electronic-health-records-for-real-world-evidence?tab=readme-ov-file.

I decided that easy functions like pivoting datasets and things like this can be written by an agent on the fly based on some agent skills. In the m4-pheno package, we should only implement complex functions that either require calculations, R code, training things, mapping with external files, etc...

# 19-03 tasks and ideas
- Explored https://github.com/aiming-lab/AutoResearchClaw/tree/main for possible ideas: the repo is a sequence of 23 STEPS that get chained one after the other, with some specific gateways for approval. The pipeline can be run with a sequence of API calls to your LLM of choice, using disk files as the memory sent from previous STEPS into context. They also include something called Agent Client Protocol that replaces the API LLM calls with an agent like Claude Code with persistent memory, and allows calling it from the python scripts. It is a very comprehensive pipeline, it covers many use cases, and is thought to be fully autonomous. It can be called from OpenClaw. It also includes modules to write a paper, turn it into latex, validate citations... It is a great source of ideas for our pipeline.
- Wrote the research skill based on the M4 skill, removing vitrine for the moment to keep it as simple as possible.
- Ran the pipeline once on MIMIC-iv-demo while the main mimic db downloads (around 12h IDK why) and making it fetch clinical notes from bigquery for the subset of patients that are identified in the Datamart. It ran okay, all modules work, but the phenotyping failed due to the small dataset.
- Explored incorporating a literature review module based on https://consensus.app/home/api/ to complement the phenotyping process and the research planning phase. The module will work as a separate subagent that is spawned by claude code so it does not clog the context, and the final summary will be written on disk so the main flow can incorporate it.
- Explored different mapping requirements in MIMIC regarding drugs, I found a few mapping dictionaries in different repos: https://github.com/OHDSI/MIMIC https://github.com/MIT-LCP/mimic-code, and I am in progress of defining a tool to bridge NDCs to RXNorm ingredients. The idea is to have as many preprocessing tools and dicts as possible + documentation of how to get them, so the agent can intake datasets that not only have flexible schemas but also flexible coding vocabs.
- Explored using vitrine as defined in M4, it could be very cool. Since it adds another layer of complexity, I kept it out of the current iteration, but left the output folder structure intact so it can be added without much hassle.

## TODOS:
- Think how to reorganize the repo so it can be a modular structure, now it is messy. I think our project needs to be the research agent, and preprocessing should be one folder, phenotyping should be another one, the literature search (if done) another... Skills tie everything together, we could devise a pipeline.md file that schematically records the decision tree that the agent needs to follow, based on the available tools.
- Ensure that the skills are as modularized as possible, keeping the main research skill lean. Right now, it is a bit to wordy.
- Ensure that the build-datamart creates an output that is not super specific for MAP, but more general. It'd be nice to move the final specific preprocessing functions to map-phenotyping.
- Test the agent on the full mimic dataset once it finishes downloading.
- Test that the notes download correctly, prompt the system to use batches to avoid memory overflows. The skill needs to be tweaked (build-datamart).
- Reproduce the RA study, and compare results (they just need to make sense, datasets are different). Think or find 3 more studies on MIMIC that involve phenotyping and reproduce. It would be cool to identify a high throughput task where this pipeline saves a lot of time, like the coding drift, and test its findings there.
- Refine the research skill to adapt to our architecture, ensuring it is comprehensive enough but not too wordy. Include https://github.com/hannesill/m4/blob/main/src/m4/skills/clinical/clinical-research-pitfalls/SKILL.md.

### If we go down the LATTE path:
- Implement a sub-agent similar to the literature
- Check data leakage if time is incorporated: ensure the notes are filtered by the date, so future information does not affect phenotyping at point x in time (a priority).

### Extra (maybe out of scope ideas)
- Finish the literature review if Consensus.app provides the API key.
- Develop an agent that acts as a medical revewer of notes, to increase number of gold labels (for example, looking for cancer progressions or other endpoints that are not codified).
- Check if ToolUniverse can provide some tools or skills that are useful, as I saw that it has many many things.
- A sub-agent that looks at the dataset in detail and writes a report about its characteristics, missingness, distributions of key variables, etc... to help the researcher and other agents understand the data they are working with.
- A self-improval system in which the agent writes down learnings from previous iterations so it can improve over time. AutoResearchClaw has something like this.

### Out of scope
- Citations/writing/formatiing papers. We focus on results.
- Reproducibility agents.
