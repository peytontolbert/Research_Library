Gap: there is no actual LLM pretraining / finetuning pipeline here: no model code, no trainer, no optimizer/schedule configs – everything assumes an external LLM.
No unified corpus builder that turns repos + papers + graphs into sharded, tokenized training datasets.
No aggressive dedup, license filtering, PII filtering, or quality filtering tuned for code+papers.
No curriculum design (e.g., easy repos/papers first, then more complex ones; mixing ratios of code vs prose).
No contrastive / retrieval‑style objectives tying repos ↔ papers (e.g., aligning code chunks with the citing paper paragraphs).
No objectives that explicitly teach the model RepoGraph / PaperGraph structure (e.g., next‑hop prediction, citation prediction, concept‑link prediction).
No temporal objectives over repo history or paper series (e.g., predict future commits, next‑paper in a line of work).
No training regime where the model is actually trained on multi‑repo / multi‑paper contexts, so it can robustly reason over 100s of GB of code+text.
No explicit tasks like “given K repos + M papers, generate an integrated design/plan,” used as supervised training.
No large‑scale, high‑precision alignment dataset (functions ↔ paper sections, experiments ↔ code that runs them).
No supervised or contrastive training that exploits those alignments to make the model “bilingual” in code‑speak and paper‑speak.
No benchmark suites for: bug‑fixing from papers, re‑implementation from methods sections, repo‑level refactors guided by literature, literature review grounded in code, etc.
No systematic metrics for graph‑aware reasoning (e.g., can the model follow RepoGraph / citation paths correctly?).