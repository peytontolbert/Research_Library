### MASTER LIST OF ALL MODELS IN YOUR AGI SUBSTRATE

Each entry is:

Model Name
Input → Target Prediction

Models are grouped by tier:

Metadata Models

Abstract-Level Models

PDF-Level Full-Paper Models

Repository Models (Code-Level)

Cross-Domain Fusion Models (PCA/RCA)

Universal Reasoning Models (Unified AGI)

Self-Play & Skill Models


TIER 1 — METADATA MODELS (GLOBAL SCIENCE GRAPH)

These operate on the 5GB metadata (1.7M entries).

1. Metadata Embedding Model

Input: title, abstract, categories, authors
Target: continuous embedding vector
(self-supervised contrastive objective)

2. Metadata Category Classifier

Input: title + abstract
Target: arXiv category label(s)

3. Citation Prediction Model

Input: title + abstract + author info
Target: list of likely citations (paper IDs)

4. Paper Similarity/Link Prediction (Graph Model)

Input: node in co-author/citation graph
Target: edges to similar papers
(graph link prediction)

5. Author Embedding / Community Model

Input: author ID
Target: embedding representing scientific field / community

─────────────────────────────────────────────
TIER 2 — ABSTRACT-LEVEL MODELS (FAST, HIGH-LEVEL SEMANTICS)
6. Abstract → Code-Relevance Classifier

Input: paper abstract
Target: which GitHub repo(s) it corresponds to
(e.g., Transformers, PyTorch, CPython)

7. Abstract → Method Summary Model

Input: abstract
Target: 2–5 sentence summary of method + contribution

8. Abstract → Paper Keywords Model

Input: abstract text
Target: auto-generated keyword list
(supervised or contrastive)

9. Abstract → Repo-Planning Model

Input: abstract + repo graph
Target: what modules/classes within repo map to the paper

─────────────────────────────────────────────
TIER 3 — FULL-PDF MODELS (500GB SCIENTIFIC CORPUS)
10. PDF Tokenization Model

Input: raw PDF
Target: structured tokens: text, equations, tables, captions, references
(OCR + math parser)

11. Full-Paper Language Model

Input: entire paper (structured tokens)
Target: next-token prediction
(LLM finetune on full papers)

12. Paper Section Predictor

Input: methods section
Target: algorithm steps in pseudo-code

13. Equation Reasoning Model

Input: mathematical equations from PDF
Target: algebraic simplification, explanation, or missing steps

14. Figure/Table Interpretation Model

Input: figure image or table snippet
Target: textual explanation / summary / extracted values

─────────────────────────────────────────────
TIER 4 — REPOSITORY MODELS (CODE-GROUNDED)
15. Repo Embedding Model (RCA Base)

Input: repository AST, functions, modules
Target: repo embedding vector (latent representation)

16. File-Level Embedding Model

Input: source file
Target: embedding representing file purpose & semantics

17. Code Q&A Model (Repo-Conditioned Adapter)

Input: (code snippet, question)
Target: answer grounded in repo structure

18. Bug Localization Model

Input: codebase + failing test description
Target: predicted file + line numbers

19. Code Mutation Model (Self-Play Friendly)

Input: code context
Target: minimal edit patch that improves correctness

20. Repo-to-Repo Similarity Model

Input: repo A, repo B
Target: similarity score for knowledge transfer

─────────────────────────────────────────────
TIER 5 — CROSS-DOMAIN MODELS (PAPER ↔ CODE UNIFICATION)

(These are the PCA/RCA “Adapter Fusion” models.)

21. Paper → Code Generator

Input: method/pseudocode section
Target: runnable Python/PyTorch implementation

22. Repo-Paper Alignment Model

Input: paper + repo
Target: mapping between paper methods and repo functions/classes

23. Paper-Conditioned Adapter (PCA)

Input: paper embedding
Target: LoRA-style delta that modulates base LLM behavior

24. Repo-Conditioned Adapter (RCA)

Input: repo embedding
Target: LoRA delta that grounds the model to that repo

25. Adapter Fusion Model (PCA ⊕ RCA)

Input: (paper embedding, repo embedding)
Target: fused latent space for cross-domain reasoning

26. Cross-Modal Retrieval Model (PDF ↔ Repo)

Input: paper or code snippet
Target: the corresponding code or paper nodes

─────────────────────────────────────────────
TIER 6 — UNIVERSAL REASONING MODELS (UNIFIED AGI)

These take your entire library (papers + metadata + repos) and fuse it.

27. Unified Knowledge Model (Core AGI Model)

Input: (paper chunk, repo context, metadata graph)
Target: multi-objective reasoning:

answer questions

propose code

reason over math

generate tests

plan experiments

summarize across sources

28. World/Planner Adapter

Input: objective description
Target: sequence of steps using other models in the substrate

29. Action Selector (PAVU Loop)

Input: current state of task (plan, code, tests)
Target: {plan_next, act, verify, update}

─────────────────────────────────────────────
TIER 7 — SELF-PLAY & SKILL MODELS (AUTONOMOUS IMPROVEMENT)
30. Skill Adapter Generator

Input: task description + environment state
Target: new LoRA adapter (skill) specialized to the task

31. Self-Play Reward Model

Input: (task, code output, test results)
Target: reward score for agent policy learning

32. Curriculum Generation Model

Input: performance profiles of agents
Target: new tasks of increasing difficulty