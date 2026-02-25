# LATTE Utility

We can use steps 1 and 2 of this paper to guide the automatization of this procedure: https://www.jmir.org/2025/1/e71873. It is a tutorial for a whole pipeline to do RWE and Target Trial Emulation (TTE) on EHR data. It also covers briefly the use of KOMAP.

0. **Preprocessing needed**
- Roll up codes: ICD to PheCode, RxNorm to ingredient-level, LOINC to test-level.
- Extract NLP features: CUIs from clinical notes using NILE (Narrative Information Linear Extraction).

1. Get gold-standard labels representing the disease outcome (See below)

2. Knowledge-guided feature selection and aggregation: via ONCE or embedding based tools like BERTs or other.

3. Constructing silver standard labels with some codes or NLP features that are highly predictive of the disease, but not perfect. This is the "silver" set.

4. Training: The inputs are quite complex, we will continue working on them.

### Gold labels

LATTE requires gold labels, only a few, but they need to be of high quality. Here are some paths we can take to get those labels:

1. **LLM-as-a-Reviewer (The Agentic Path):**
An agent uses M4’s `search_notes` to pull all clinical narratives for a potential case. It then provides these notes to a "Clinician Agent" (GPT-4o or similar) with a specific prompt: *"Based on these 5 nursing notes and the discharge summary, did this patient experience a new MS relapse between Jan and March?"*


2. **Registry/Database Linkage:**
Simplest case, the database where the system is used already contains a set of gold labels in the form of verified relapses, manual curation, a link to a registry...

3. **The "Expert-in-the-Loop" (Hybrid):**
The agent performs the initial extraction and "pre-labels" the charts using the LLM method above. It then presents only the "Low Confidence" or "Conflicting" cases to a human doctor via a simple UI. Could be an interesting side quest to design, or maybe leverage something like Label Studio or Vitrine.