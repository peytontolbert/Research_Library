### MirrorMind scaffolding

This directory provides a minimal, runnable shell of the MirrorMind architecture described in `models/paper.md`:

- `memory.py` – episodic and semantic memory primitives (append-only stores + heuristic retrieval; JSONL save/load for persistence; optional dense/sparse/BM25/FAISS indexes and embedding builder in `embeddings.py`, env override `MIRRORMIND_EMBED_MODEL`; query filters for types/time_range/type_weights).
- `persona.py` – persona schema builder from exported concepts with style heuristics and prompt rendering.
- `domain.py` – DomainGraph built from `models/exports/repo_concepts.jsonl` (+ optional paper_concepts/paper_repo_align) plus DomainAgent tool APIs and pluggable graph clients (file/Neo4j; dotenv-supported).
- `twins.py` – RepoTwin and PaperTwin wrappers bundling persona, semantic, and episodic memory.
- `context.py` – context assembly for RepoTwin and PaperTwin (persona + semantic + episodic + task).
- `coordinator.py` – coordinator and review stubs that route tasks through DomainAgent and twins.
- `llm.py` – helper to load the default Llama 1B (`meta-llama/Llama-3.2-1B`) from `/data/checkpoints` for use by PIANO/Talking modules.
- `scripts/build_indexes.py` – builds dense/sparse/FAISS indexes from an episode JSONL.
- `scripts/build_semantic_memory.py` – aggregates episodic JSONL into semantic summaries (pluggable summarizer hook).
- `scripts/build_semantic_from_repo_chunks.py` – convenience to turn repo_chunks exports into per-repo semantic summaries (no sampling; uses all chunks) with optional heuristic/LLM summarization and raw context preserved.
- `scripts/extract_paper_concepts.py` – heuristic paper concept extractor from arXiv manifest into `models/exports/paper_concepts.jsonl`.
- `scripts/check_neo4j.py` – simple connectivity check using env/dotenv (NEO4J_URI/USER/PASSWORD).
- `scripts/check_neo4j.py` – simple connectivity check using env/dotenv (NEO4J_URI/USER/PASSWORD).

The goal is to expose the variables and flows required by `models/paper.md` without heavy dependencies. The classes intentionally use the existing exports (concepts) and shared data loaders as defaults and degrade gracefully when data is missing. Extend the stores with real embeddings/indexes and wire the coordinator to the 32 model stubs when ready.
