## Unified Library TODO

High-level plan to integrate the local ArXiv metadata into the existing
repository library in a **model-based** way, ignoring PDFs for now.

### 1. Clarify current layout and responsibilities

- **Inventory current modules**
  - **Repos**: `scripts/repo_library.py`, `scripts/library_repo_graph_export.py`,
    `modules/vector_index.py`, `scripts/skill_build.py`, `modules/qa_runtime.py`,
    `modules/qa_swarm.py`, `run.py` HTTP APIs.
  - **ArXiv metadata**: `/data/arxiv` snapshot,
    `modules/arxiv_library.py`, `docs/arxiv_metadata.md`,
    `/api/arxiv/search` in `run.py`.
- **Define library entities**
  - **Repository**: existing manifest/graph representation.
  - **Paper**: ArXiv metadata record (id, title, abstract, authors, categories, year, ...).
  - **Future**: full paper chunks (PDF-derived) as a separate, deeper layer.

### 2. Embedding and index strategy over metadata

- **Choose embedding model**
  - **Default**: `all-MiniLM-L6-v2` from `model.yml` (already wired via `modules/embeddings.py`).
  - Keep consistent with repo embeddings so repos and papers share a latent space.
- **Build ArXiv metadata index**
  - **Task**: offline job to embed `title + abstract + categories` for all papers.
  - **Store**:
    - Dense vectors (float16) under something like `/data/arxiv/indices/metadata/`.
    - Metadata table (Parquet/DuckDB/JSONL) with IDs and fields.
  - **Index type**:
    - Start with a simple FAISS/HNSW or a `SimpleNumpyRepoIndex`-style implementation specialized to ArXiv.
- **APIs**
  - Add a thin `ArxivIndex` abstraction that mirrors `SimpleNumpyRepoIndex`
    but lives over paper metadata instead of code entities.

### 3. Unifying repos + ArXiv in the “library” view

- **Shared embedding space**
  - **Task**: ensure repo-level descriptions (README/docs) are embedded with the same model as ArXiv.
  - **Functionality**:
    - `related_papers_for_repo(repo_id)`:
      - Embed repo description.
      - Query ArXiv metadata index.
    - `related_repos_for_paper(paper_id)`:
      - Embed paper metadata.
      - Query repo index.
- **Library-level abstraction**
  - **Task**: introduce a light wrapper (e.g. `UnifiedLibrary` or new methods on `RepoLibrary`) that:
    - Can return both **repo entries** and **paper entries** as first-class objects.
    - Exposes cross-links (repo ⇄ paper) via the shared embedding space.

### 4. Model-based skills over ArXiv metadata

- **New skill type**
  - **Task**: define an initial `"arxiv_qa"` or `"arxiv_search"` skill that:
    - Takes a natural language query.
    - Uses the ArXiv metadata index as retrieval.
    - Uses the existing LLaMA QA runtime to:
      - Summarize top-k papers.
      - Generate reading lists.
- **Adapter schema**
  - **Extend** the adapter/registry schema to:
    - Allow non-repo skills (e.g. `"arxiv_qa"` with no single `repo_id`).
    - Or treat ArXiv as a synthetic “repo” entry in `_manifest.json`.
- **Routing**
  - **Task**: update `SemanticRouter` so it can:
    - Route some questions to ArXiv-only skills.
    - Mix “repo QA” with “suggest relevant papers” as separate steps.

### 5. Swarm / planner integration (metadata only)

- **Planner**
  - **Task**: extend `QASwarmController` / router to support:
    - A multi-step flow: repo QA → ArXiv retrieval → combined answer.
    - Or a dedicated “paper suggestion” interaction mode.
- **Retriever agents**
  - Add a dedicated `ArxivRetrieverAgent` that:
    - Calls the ArXiv metadata index / `/api/arxiv/search`.
    - Returns structured paper snippets (id, title, abstract, categories).
- **Context shaping**
  - Define a simple context template for LLM over metadata:
    - Sections like “Top candidate papers”, “Reasons they are related”.

### 6. Evaluation, logging, and iteration

- **Logging**
  - Log ArXiv search calls:
    - Query, filters, selected papers, downstream usage (e.g. clicked/opened).
- **Quality checks**
  - Manual spot checks:
    - Given a repo, do related papers make sense?
    - Given a query, are top-10 papers relevant?
- **Future adapter training**
  - Use logs to:
    - Train cross-encoders / ranking adapters over paper metadata.
    - Train LoRA skill adapters specialized for:
      - Reading list generation.
      - Meta-summaries of clusters of papers.

### 7. Out of scope for this phase (tracked separately)

- **PDF / full-text integration**
  - Chunk extraction, passage-level embedding, and deep QA over paper bodies.
  - “PaperReader” / “MethodExtractor” / “CodeLinking” adapters.
  - These should be defined in a separate TODO once the metadata pipeline and
    unified library are solid.


## Program Graph Full-Coverage Upgrade

High-level, incremental plan to evolve the program graph from Python-only
structure to a broader, multi-language, file-aware view suitable for large
repos like `linux`, `transformers`, `cpython`, `v8`, `torch`, etc.

### Phase 0 — Baseline documentation and invariants

- **Document current graph behavior**
  - Capture how `CodeGraph`, `PythonRepoGraph`, and `RepoGraph` work today:
    - Python-only AST parsing for modules/classes/functions/tests.
    - Edges: `owns`, `imports`, `calls`, `tests`.
    - Artifacts: all repo files as `Artifact(type="source")` with hashes.
  - Clarify how exports are wired:
    - `scripts/library_repo_graph_export._export_repo_graph`.
    - `modules/vector_index.build_repo_qa_index` (entity selection).
- **Define invariants to preserve**
  - Existing Python repos’ behavior must remain backwards compatible:
    - Same or strictly superset of entities/edges.
    - Same `program://` URI scheme for Python entities.
  - Export format (`*.entities.jsonl`, `*.edges.jsonl`, `*.artifacts.jsonl`)
    should remain compatible; only additive changes allowed.

### Phase 1 — File-level entities for all source files

- **Introduce file entities**
  - Define a new `Entity` kind `file`:
    - `id = "file:{rel_path}"` (Unix-style path, repo-relative).
    - `uri = "program://{program_id}/file/{rel_path}"`.
    - `name = rel_path` (or basename); `labels` may include coarse language tags.
  - Implement a helper on `RepoGraph` or a small utility to:
    - Walk `_discover_files` and build file entities with hashes.
- **Integrate into `PythonRepoGraph.entities()`**
  - Keep existing Python entities (modules/functions/classes/tests) unchanged.
  - Add file entities for:
    - All non-`.py` files.
    - Optionally also `.py` files:
      - Either skip them (module entity is enough), or
      - Add a `file` entity plus an `owns` edge from file → module.
  - Ensure entity IDs remain globally unique within the repo.
- **QA index inclusion strategy**
  - Decide which entity kinds feed into `build_repo_qa_index`:
    - Option A: include `file` entities for all languages (simple, more recall).
    - Option B: include only some `file` kinds (C/C++/JS/MD) with labels.
  - Update `build_repo_qa_index` to:
    - Either filter by `kind` (e.g., `{"module","class","function","file"}`).
    - Or at least not break when new kinds appear.
- **Validation**
  - Export a small repo and inspect:
    - `*.entities.jsonl` includes `file` entities for C/C++/JS/MD/etc.
    - QA index builds successfully and includes a mix of entity kinds.

### Phase 2 — Language-aware tagging for files

- **Lightweight language detection by extension**
  - Define a simple mapping from file extension → language tag, e.g.:
    - `.c`, `.h`, `.cc`, `.cpp`, `.cxx` → `lang:c`, `lang:cpp`.
    - `.js`, `.jsx`, `.mjs` → `lang:js`.
    - `.ts`, `.tsx` → `lang:ts`.
    - `.py` → `lang:python`.
    - `.md` → `lang:markdown`.
  - Attach these tags to `file` entities via `labels` or a prefix convention.
- **Update exporters and tooling expectations**
  - Document in the design doc how downstream tools should interpret:
    - `kind="file"` + `labels=["lang:c"]` etc.
  - Ensure nothing breaks if labels are absent (backwards compatibility).

### Phase 3 — Basic structure for non-Python languages (C/C++/JS)

- **Scope and constraints**
  - Target only: C/C++ and JS/TS for this phase.
  - Rust and more exotic languages are explicitly out-of-scope.
- **Per-language mini analyzers**
  - Introduce simple analyzers to extract:
    - Functions and (optionally) types/classes for:
      - C/C++ (`*.c`, `*.h`, `*.cc`, `*.cpp`, `*.cxx`, `*.hpp`).
      - JS/TS (`*.js`, `*.jsx`, `*.ts`, `*.tsx`).
  - Start with minimal goals:
    - Identify top-level function names with approximate spans.
    - Optionally infer simple `owns` edges: file → function.
  - Consider Tree-sitter or other lightweight parsers, but keep the surface small:
    - A small wrapper per language that returns `Entity` + `Edge` sets.
- **Integrate into a multi-language RepoGraph**
  - Either:
    - Extend `PythonRepoGraph` to call per-language analyzers alongside `CodeGraph`, or
    - Introduce `MultiLangRepoGraph` that composes:
      - Python `CodeGraph`.
      - C/C++ analyzer.
      - JS/TS analyzer.
  - Merge:
    - Entities: Python entities + C/C++/JS/TS function entities + file entities.
    - Edges: `owns` (file→function/module), plus any cheap call/import edges if available.
- **Export and QA index updates**
  - Ensure `library_repo_graph_export` still just calls `graph.entities()` and `graph.edges()`.
  - QA index:
    - Include non-Python function entities with kind tags like `c_function`, `js_function`, etc., or reuse `function` with language in `labels`.

### Phase 4 — Optional: commit-aware metadata (not full commit graph)

- **Keep commits out of ProgramGraph for now**
  - Commits remain metadata (manifest + API fields like `commit_count`).
  - Do not introduce `Entity(kind="commit")` yet to keep the graph small.
- **Enrich manifest-level Git metadata**
  - Optionally store:
    - Last N commits’ SHAs + timestamps + authors per repo.
    - Simple churn stats per repo (e.g., number of commits in the last 30 days).
  - Expose via `/api/repos/{repo_id}` for UI and analytics.

### Phase 5 — Documentation and rollout

- **Dedicated design doc**
  - Add `docs/program_graph_full_coverage.md` (or similar) with:
    - Current vs. target architecture diagrams.
    - Entity/edge/kind tables (including `file` and language labels).
    - Examples from a large repo (e.g., linux, torch, transformers).
  - Keep it updated as phases complete.
- **Incremental rollout strategy**
  - Enable new graph behavior behind a flag or schema version bump:
    - Allow exporting a single repo with “new graph” for validation.
  - Rebuild exports for a small set of target repos first (linux, cpython, torch).
  - Once stable, run a full library export.



