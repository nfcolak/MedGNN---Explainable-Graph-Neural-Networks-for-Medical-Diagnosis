# MSc Thesis Proposal 5 — Static vs. Agentic RAG for Industrial Technical Documentation

*Evidence-based proposal, built on verified completed MSc theses. Prepared 2026-07-10.*

---

## Part 1 — Verified foundation evidence (completed MSc theses)

**Transparency note on rankings.** The foundation thesis (KTH) is from a university consistently ranked within the global top 100 (QS World University Rankings, top ~75 in recent editions). Two of the strongest *content* matches (Bauhaus-Universität Weimar, Luleå University of Technology) are from institutions outside the overall top 100 and are used as feasibility co-evidence, not as the primary foundation. Aalto University sits slightly outside the overall QS top 100 but within the top 100 for Engineering & Technology subject rankings. No directly matching top-100 thesis on *agentic* RAG specifically was found — the KTH thesis is the closest verifiable top-100 anchor, and this is stated openly rather than over-claimed.

### E1 (Foundation) — Beyond the Basics: Advanced Retrieval Techniques for RAG Systems
- **Author:** Fred Olsson
- **University:** KTH Royal Institute of Technology, School of Electrical Engineering and Computer Science (EECS)
- **Year:** 2024 — Master's thesis (Two Years), 30 credits
- **Source:** DiVA record `diva2:1918448` — https://kth.diva-portal.org/smash/record.jsf?pid=diva2:1918448 (full text: https://kth.diva-portal.org/smash/get/diva2:1918448/FULLTEXT01.pdf)
- **Research problem:** Do advanced retrieval techniques (sentence-window retrieval, auto-merging retrieval) measurably improve RAG performance in a real company setting (Swedish consulting firm)?
- **Methodology:** Controlled experiments with difficulty-stratified question sets; metrics: faithfulness, answer relevancy, context relevancy; statistical significance testing; complemented by interviews.
- **Main findings:** No statistically significant improvement from advanced static retrieval in most metrics; auto-merging improved faithfulness only for medium-difficulty questions; baseline sometimes better on context relevancy.
- **Adaptable elements:** the entire controlled comparative design — difficulty/type-stratified question sets, RAGAS-style metric suite, significance testing, industrial single-company corpus setting.

### E2 — Multi-Agent Retrieval-Augmented System for Domain-Specific Knowledge in Structural Engineering
- **Author:** Hafiz Muhammad Ahmad
- **University:** Aalto University, Master's Programme in Building Technology (supervisor: Asst. Prof. Mohamed Noureldin; collaborative partner: Ramboll)
- **Year:** 2025 (29.09.2025), 65 pages — Master's thesis
- **Source:** Aaltodoc — https://aaltodoc.aalto.fi/bitstreams/9be4c3e5-1513-4869-94f1-1f135838ebc8/download
- **Research problem:** LLMs fail on country-specific engineering code provisions (Finnish National Annexes to the Eurocode); can an agentic RAG assistant give code-compliant, grounded answers?
- **Methodology:** Two-agent RAG (agent 1: clause-level retrieval from annex corpus; agent 2: fallback to general Eurocode knowledge); evaluation on real design-scenario queries across model settings.
- **Main findings:** RAG answers were accurate and grounded in annex text; two-agent design guarantees a response for every query; smaller LLMs eliminated hallucinations at the cost of partial answers; larger models gave more complete answers but occasionally introduced unsupported details.
- **Adaptable elements:** agentic/tool-routing RAG design with fallback logic; clause-level traceability focus; hallucination-vs-completeness trade-off analysis; proof that agentic RAG over engineering standards is one-student feasible.

### E3 — Retrieval Augmented Generation for Industrial Documentation
- **Author:** Mohamed Salama
- **University:** Bauhaus-Universität Weimar, Faculty of Media, Degree Programme Digital Engineering (referees: Prof. Dr. Benno Stein — Webis IR group; Prof. Dr. Andreas Jakoby); industry partner: Dieffenbacher GmbH
- **Year:** 2025 (submitted 19.03.2025) — Master's thesis
- **Source:** Webis theses — https://downloads.webis.de/theses/papers/salama_2025.pdf
- **Research problem:** Extracting actionable answers from heterogeneous industrial documentation (multilingual text, structured tables, technical diagrams).
- **Methodology:** RAG over 423 real industrial documents; LlamaIndex; multilingual-E5-large vs. ada-002 embeddings; Llama 3 / Llama 3.1 / Qwen2 (incl. quantization); dense vs. sparse vs. hybrid retrieval; SQL-based retriever for tabular data; an agent-based extension; dual-phase evaluation (automated LLM judging + manual/expert evaluation, incl. tabular questions and technical-terminology analysis).
- **Main findings:** Demonstrated a working industrial-documentation RAG; identified concrete failure modes on tables and domain terminology; documented challenges/solutions for industrial RAG pipelines; agent-based solution explored only preliminarily.
- **Adaptable elements:** heterogeneous-document ingestion (text + tables), hybrid retrieval baseline, dual-phase (automated + expert) evaluation protocol, tabular-question stratum.

### E4 (supporting) — Evaluating Retrieval-Augmented Generation Architectures for Single and Multi-Hop Question Answering
- **Author:** Stepan Cherevichnik — Luleå University of Technology, Master Programme in Applied AI, 2025 — Master's thesis (Two Years), 30 credits
- **Source:** DiVA record `diva2:2002498` — https://www.diva-portal.org/smash/record.jsf?pid=diva2:2002498
- **Problem/method:** Compared dense, hybrid, query-expansion, and HyDE RAG pipelines on single- vs. multi-hop QA; built a synthetic multi-hop QA dataset via LLM-generated persona queries over a scraped document collection; metrics: MRR, faithfulness, context relevance, semantic similarity.
- **Findings:** Retrieval quality dominates downstream generation performance, especially for multi-hop queries.
- **Adaptable elements:** semi-automatic, human-verified QA dataset construction; single- vs. multi-hop stratification.

### E5 (supporting, organizational relevance) — AI Adoption in Research & Innovation
- **Author:** Patricia de Frutos Pérez — KTH, School of Industrial Engineering and Management, 2025 — Master's thesis, 30 credits (case: Scania R&I Office)
- **Source:** DiVA record `diva2:2002017` — https://www.diva-portal.org/smash/record.jsf?pid=diva2:2002017
- **Relevance:** Qualitative evidence (5-month ethnographic case study) that RAG-based knowledge accessibility is a recognized industrial-management problem — supports the Industrial Engineering framing, not the technical design.

---

## Part 2 — Foundation selection

**Selected foundation: E1 — Olsson (KTH, 2024).**

**Why:** It is the strongest verifiable top-100 anchor with a cleanly documented, directly reusable research design: a controlled comparison of retrieval strategies in one industrial organization, stratified questions, standard RAG quality metrics, and statistical testing. Crucially, its *null result* — advanced **static** retrieval yielded no significant gains — poses exactly the question this proposal answers next: does **agentic** retrieval (dynamic, decision-making retrieval) deliver measurable gains where static sophistication did not, and at what cost? E2 (Ahmad/Aalto) proves agentic RAG with traceability focus is MSc-feasible; E3 (Salama/Weimar) proves heterogeneous industrial-document RAG with dual-phase evaluation is MSc-feasible.

- **Replicated/adapted (from E1):** controlled single-corpus comparative design; stratified question set; faithfulness/relevancy metric suite; significance testing. From E2: agentic tool-routing architecture and attribution focus. From E3: heterogeneous corpus handling (text + tables) and automated + expert dual-phase evaluation. From E4: semi-automatic QA-set construction.
- **Changed:** the comparison axis moves from static-retrieval variants (sentence-window vs. auto-merging) to **static RAG vs. agentic RAG**; the corpus moves to heterogeneous industrial technical documentation; the metric suite is extended with attribution accuracy, unanswerable-question handling, and cost/latency.
- **New academic contribution:** the first controlled, statistically tested, *cost-aware and question-type-conditioned* comparison of static vs. agentic RAG on heterogeneous industrial technical documentation — producing evidence on *when agency pays off*, not merely another system demo.

---

## Part 3 — Final proposal (≤ 2 pages)

### Working Title
**When Does Agency Pay Off? A Controlled Comparison of Static and Agentic Retrieval-Augmented Generation for Question Answering over Heterogeneous Industrial Technical Documentation**

### Background and Motivation
Industrial and engineering organizations accumulate large volumes of heterogeneous technical documentation — operation and maintenance manuals, standards, procedures, and reports mixing prose with tables and domain terminology. Engineers spend substantial time locating trustworthy answers; keyword search is insufficient, and standalone LLMs hallucinate. Static RAG grounds answers in documents but runs a fixed retrieve-then-generate pipeline regardless of question complexity. Agentic RAG lets an LLM agent decide *when, what, and how* to retrieve — decomposing questions, choosing between text and table retrieval, iterating, and self-verifying — but adds latency, cost, and unpredictability. Organizations currently lack controlled evidence on whether, and for which questions, this added complexity is justified.

### Research Gap
Olsson (KTH, 2024) showed that advanced *static* retrieval produced no statistically significant gains in an industrial RAG setting — leaving open whether *adaptive/agentic* retrieval does better. Salama (Weimar, 2025) built RAG over 423 real industrial documents and identified tables and terminology as failure modes, but explored the agent-based solution only preliminarily. Ahmad (Aalto, 2025) demonstrated an agentic RAG over engineering codes with strong groundedness, but compared model settings rather than paradigms and reported no cost analysis. Surveys of agentic RAG (Singh et al., 2025) and an industry interview study on RAG (Bogner et al., 2025, arXiv:2508.14066) both flag the absence of systematic, evaluation-driven comparisons in industrial settings. **Gap:** no controlled, statistically tested comparison of static vs. agentic RAG on heterogeneous industrial technical documentation that jointly measures answer quality, traceability, refusal behavior, and efficiency.

### Research Objective
Design and implement one static RAG pipeline and one agentic RAG system over the same heterogeneous industrial technical-document corpus, and evaluate them under controlled conditions across question types, quantifying quality (correctness, groundedness, attribution), safety behavior (unanswerable-question handling), and efficiency (latency, tokens, cost) — including component ablations to attribute observed differences.

### Research Questions
- **RQ1 (primary):** To what extent, and for which question characteristics (single-hop, multi-hop, tabular, unanswerable), does agentic RAG improve answer correctness and groundedness over static RAG on heterogeneous industrial technical documentation, and at what cost in latency and token usage?
- **RQ2:** Which agentic components (iterative retrieval, tool selection between text and table retrievers, self-verification) contribute most to the observed differences?
- **RQ3:** How do the two paradigms differ in attribution accuracy (traceability of answers to source passages) and in refusing unanswerable questions?

### Proposed Methodology
- **Corpus:** 200–500 publicly available industrial technical documents (e.g., public equipment operation & maintenance manuals, openly accessible engineering standards/guidelines, safety documentation), deliberately heterogeneous (prose + tables). A public technical-QA benchmark (e.g., TechQA or FreshStack) serves as a secondary corpus for external validity. No proprietary data is required; an optional internal validation at the collaborating company is a bonus, not a dependency.
- **Question set:** 150–300 questions, stratified into single-hop factual, multi-hop/cross-document, table/numerical, and unanswerable; generated semi-automatically (LLM-drafted, fully human-verified; domain-expert check on a subset), following Cherevichnik (2025) and Salama (2025).
- **Systems:** **B0** closed-book LLM (no retrieval); **B1** static RAG: hybrid retrieval (BM25 + dense) + reranking + single generation with citation prompting; **A1** agentic RAG: one ReAct-style orchestrator with tools — hybrid text retriever, table retriever, query decomposition, retrieval-sufficiency self-check, answer-or-refuse. Two ablations of A1 (without iteration; without self-verification). One GPT-4-class API model (e.g., via Azure OpenAI) as primary, one smaller model for robustness. Frameworks: LlamaIndex or LangGraph + a standard vector database.
- **Metrics:** retrieval Recall@k and MRR; answer correctness (LLM-as-judge validated against human ratings on a subset, with inter-rater agreement); faithfulness and context relevance (RAGAS-style); attribution precision/recall (does the cited passage actually support the answer?); refusal accuracy / false-answer rate on unanswerable questions; efficiency (LLM calls, tokens, latency, cost per query).
- **Analysis:** paired significance tests (Wilcoxon signed-rank / bootstrap) per question stratum, following Olsson (2024); quality–cost trade-off curves; error taxonomy.
- **Qualitative:** small expert review — 3–5 engineers rate ~30 stratified answers (dual-phase protocol from Salama, 2025).

### Expected Contribution
- **Academic:** controlled, statistically grounded evidence on *when* agentic RAG outperforms static RAG on industrial technical documents, conditioned on question type — extending Olsson's static-retrieval null result to the agentic axis and answering the comparison gap flagged in the agentic-RAG literature.
- **Technical:** a reusable, open evaluation harness: corpus-preparation pipeline for heterogeneous documents, stratified QA-generation protocol, attribution and refusal metrics, and per-query cost accounting, plus reference implementations of both paradigms.
- **Practical:** evidence-based decision guidance (quality-vs-cost) for any documentation-heavy engineering organization choosing between static and agentic RAG — directly applicable to maintenance, compliance, and operational knowledge workflows.

### Scope and Limitations
**In scope:** one primary public corpus + one benchmark; three systems + two ablations; two LLMs; text and tables; English; automated evaluation with human-verified subsets; small expert review. **Out of scope:** model fine-tuning or training; images/diagrams and multimodal retrieval; multilingual evaluation; multi-company generalization studies; production deployment; large user studies; enterprise integration.

### Feasibility
All data is public; no proprietary corpus is required. No model training occurs — only API inference (Azure OpenAI-class access is available in the collaborating environment) and open-source retrieval components on a laptop/modest VM. The experimental workload is bounded: ≤300 questions × ~6 system variants × 2 models ≈ a few thousand LLM calls, well within a normal thesis budget. Mature frameworks (LlamaIndex/LangGraph, standard vector DBs, RAGAS) minimize engineering risk. Each methodological element has already been executed by a single MSc student in a verified completed thesis (E1–E4), which is the strongest available feasibility evidence. Estimated effort fits a standard 30 ECTS / ~6-month thesis.

### Preliminary Work Plan
- **P1 (weeks 1–6):** literature review; corpus assembly and ingestion; question-set design and human verification.
- **P2 (weeks 5–10):** static RAG baseline (B0, B1); evaluation harness incl. judge validation.
- **P3 (weeks 9–16):** agentic system (A1) and ablations.
- **P4 (weeks 15–20):** full experimental runs; statistical analysis.
- **P5 (weeks 19–22):** expert review; error taxonomy.
- **P6 (weeks 21–26):** synthesis, trade-off analysis, thesis writing and defense preparation.

### Key References
1. Olsson, F. (2024). *Beyond the Basics: Advanced Retrieval Techniques for RAG Systems.* MSc thesis, KTH Royal Institute of Technology. DiVA: diva2:1918448. **(foundation thesis)**
2. Ahmad, H. M. (2025). *Multi-Agent Retrieval-Augmented System for Domain-Specific Knowledge in Structural Engineering.* MSc thesis, Aalto University (Aaltodoc).
3. Salama, M. (2025). *Retrieval Augmented Generation for Industrial Documentation.* MSc thesis, Bauhaus-Universität Weimar (Webis theses).
4. Cherevichnik, S. (2025). *Evaluating Retrieval-Augmented Generation Architectures for Single and Multi-Hop Question Answering.* MSc thesis, Luleå University of Technology. DiVA: diva2:2002498.
5. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020.*
6. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023.*
7. Es, S., et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024 (System Demonstrations).*
8. Singh, A., et al. (2025). Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG. *arXiv:2501.09136.*
9. Gao, Y., et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997.*
10. Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation. *arXiv:2508.14066* (2025).
