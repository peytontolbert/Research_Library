Repository Library
==================

A backend service and knowledge system that indexes code repositories stored in `/data/repositories` plus any configured extension roots, builds a graph-centric representation of them under `/data/repository_library/exports`, and (in later phases) exposes LLM-powered retrieval and editing over this **repository library**.

The current implementation focuses on:

- Exporting a **per-repository program graph** as JSONL.
- Maintaining a **library manifest** (`exports/_manifest.json`) with lightweight per-repo metadata.
- Providing the foundations for **Repository objects** and **SkillSets** that higher-level services (Q&A, refactors, meta-learning) can use.
- Exposing those capabilities to agent clients through a stdio MCP server.

### Goals

- **Central library of repositories**: Treat `/data/repositories` plus configured extension roots as a single searchable "library" of code and related assets.
- **Graph-based knowledge model**: Represent repositories, files, symbols, and relationships (dependencies, ownership, references) as a program graph.
- **Repository-centric skills**: For each repository, attach a set of *skills* (adapters, tools, indices) that specialize the system for Q&A, editing, navigation, testing, etc.
- **LLM-powered workflows** (later phase): Answer questions, propose edits, and distill meta-knowledge over the entire library.

### Repository-Centric View

Conceptually, each repository is normalized into a `Repository` object:

```python
class Repository:
    repo_id: str          # e.g. "linux" or "github:python/cpython"
    root_path: Path       # on-disk mirror under a configured library root
    metadata: dict        # tags, languages, git state, etc.
    graph: ProgramGraph   # code structure, symbols, dependencies
    index: RepoIndex      # text/code embeddings, search indices
    tools: RepoTools      # repo-scoped tools (grep, tests, build, etc.)
    skills: SkillSet      # repo-conditioned adapters / skills
```

The **export pipeline** in `build.py` and `scripts/library_repo_graph_export.py` is responsible for:

- Discovering repositories under `/data/repositories` and any persisted extension roots.
- Building a `PythonRepoGraph` per repo.
- Writing JSONL exports:
  - `{repo_id}.entities.jsonl`
  - `{repo_id}.edges.jsonl`
  - `{repo_id}.artifacts.jsonl`
- Updating a shared manifest in `exports/_manifest.json` with per-repo metadata and (in later steps) skillset information.

### Public Datasets Created From This Repo

This repository also contains the scripts used to publish public Hugging Face datasets derived from those exports.

#### `PeytonT/repo_graph`

Dataset: [huggingface.co/datasets/PeytonT/repo_graph](https://huggingface.co/datasets/PeytonT/repo_graph)

Relevant scripts:

- `scripts/library_repo_graph_export.py` builds the per-repo graph exports under `exports/`.
- `scripts/export_library_repo_graph_hf_dataset.py` converts those exports into a Hugging Face-ready parquet dataset and dataset card.

What the dataset is:

- A parquet-first snapshot of the repository library's program graph.
- It preserves both the per-repository graph exports and the aggregated universe graph.
- The published configs are `repos`, `entities`, `edges`, `artifacts`, `universe_nodes`, `universe_edges`, and `repo_knn`.
- `repos` contains repository metadata plus optional 3D universe coordinates.
- `entities` contains graph nodes such as files, modules, classes, and functions.
- `edges` contains intra-repo graph relationships such as ownership and references.
- `artifacts` contains artifact URIs and hashes, not full file contents.
- `universe_nodes` and `universe_edges` capture the cross-repository universe projection.
- `repo_knn` is a repository-to-repository nearest-neighbor graph with edge weights.

Current public snapshot:

- As published on April 19, 2026, the public dataset contains `15,499,276` total rows across all configs.
- That includes `178` repo rows, `906,024` entities, `8,402,561` per-repo edges, `453,278` artifacts, `216,987` universe nodes, `5,518,898` universe edges, and `1,350` repo-to-repo KNN edges.
- In practical terms, it is a large machine-readable map of the code library rather than a source-code dump.

#### `PeytonT/1m_papers_text`

Dataset: [huggingface.co/datasets/PeytonT/1m_papers_text](https://huggingface.co/datasets/PeytonT/1m_papers_text)

Relevant scripts:

- `scripts/export_paper_text_hf_dataset.py` emits one row per paper and publishes the Hugging Face dataset.
- `scripts/backfill_missing_paper_text_shards.py` fills in missing full-text rows from local arXiv PDFs.
- `scripts/backfill_paper_text_from_gcs.py` grows the corpus by temporarily downloading missing PDFs from GCS and writing parquet backfill shards.
- `scripts/distributed_paper_text_backfill.py` coordinates that same backfill workflow across multiple workers.
- `scripts/merge_paper_text_parquets.py` merges base exports plus backfills and dedupes on `canonical_paper_id`.
- Operator guide: [docs/paper_text_backfill_cluster.md](docs/paper_text_backfill_cluster.md)

What the dataset is:

- A public full-text paper dataset with one row per arXiv paper.
- Each row includes `paper_id`, `canonical_paper_id`, `paper_version`, `pdf_path`, `title`, `abstract`, `authors`, `categories`, `license`, and the extracted `text`.
- The export also records provenance fields such as `text_source`, `text_is_partial`, `text_char_count`, `page_count`, `token_count`, and `token_types`.
- The key idea is that the dataset is not just raw text; it keeps enough metadata to trace where the text came from and whether it was reconstructed from structured tokens or extracted directly from PDFs.

Current public snapshot:

- As published on April 19, 2026, the public dataset contains `1,000,000` papers, all with matched arXiv metadata.
- The export was generated with `--prefer-raw-pdf-text` and `--raw-pdf-max-chars 0`, which means the PDF-sourced rows are intended to be full-document extracts rather than capped snippets.
- In practical terms, this dataset is a public text corpus for paper-scale retrieval, search, and training workflows, with enough metadata to filter by license and provenance before reuse.

### SkillSets (Per-Repository Skills)

On top of the graph and indices, each repo can expose a `SkillSet` – a collection of adapters and tools specialized for different workflows:

```python
class SkillSet:
    qa: QAAdapter            # repo-aware Q&A over code and docs
    edit: EditAdapter        # safe refactors & edits in repo style
    meta: MetaAdapter        # distilled patterns, APIs, idioms
    nav: NavAdapter          # intent → graph navigation
    test: TestAdapter        # test discovery and generation
    perf: PerfAdapter        # performance-oriented reasoning
    security: SecurityAdapter# security patterns for this repo
    api: APIAdapter          # public API surface & changes
    style: StyleAdapter      # naming / formatting / idioms
```

In the current codebase:

- The **graph exports** are already implemented.
- The **SkillSet layer** is documented and sketched in code, but actual adapters (weights, runtime integration) are intentionally left as future work.

### High-Level Architecture (Draft)

- **Ingestion & Export**
  - Scans the default repository root and any persisted extension roots on a schedule or trigger.
  - For each repo, builds a `PythonRepoGraph` (see `scripts/python_repo_graph.py`).
  - Writes entities/edges/artifacts as JSONL under `/data/repository_library/exports/{repo_id}`.
  - Maintains a library-level manifest in `exports/_manifest.json`.

- **Graph & Index Layer** (this repo)
  - Defines minimal `ProgramGraph` interfaces (`modules/program_graph.py`).
  - Provides utilities for per-repo graphs (`scripts/repo_graph.py`, `scripts/python_repo_graph.py`).
  - Will later expose library-level helpers to load `Repository` objects from exports.

- **LLM & Skills Layer** (later phase, external service)
  - Loads `Repository` objects (graph + indices + skills).
  - Dispatches user queries/tasks to appropriate skills:
    - Q&A, refactor, meta-learning, benchmarking, etc.
  - Runs on top of `meta-llama/Llama-3.1-8B-Instruct` (or similar) with RAG over the library.

### Phases

1. **Phase 1: Graph Exports (current)**
   - Export per-repo entities, edges, and artifacts as JSONL.
   - Maintain `_manifest.json` with basic per-repo metadata.

2. **Phase 2: Repository Objects + SkillSet Skeleton**
   - Implement a lightweight in-process `Repository` abstraction that loads from exports.
   - Add a `SkillSet` skeleton (QA, edit, meta, nav, test, perf, security, api, style).
   - Expose a simple Python API for algorithms to open a repo by `repo_id`.

3. **Phase 3: Indices & Adapters**
   - Build `RepoIndex` (embeddings + search indices) from exports.
   - Train or plug in per-repo adapters for key skills (QA, edit, meta).
   - Extend `_manifest.json` to record available skills and adapter artifacts.

4. **Phase 4: Services & UI**
   - Provide HTTP APIs for Q&A, search, and agentic editing over the library.
   - Add a minimal UI or CLI for interactive exploration.

For deeper architecture details and design decisions, see `ARCHITECTURE.md`.

### Agent Access

Repository-aware agents can connect through the stdio MCP server:

```bash
PYTHONPATH=/data/repository_library /home/peyton/miniconda3/envs/ai/bin/python -m scripts.repo_library_mcp
```

There is also a helper to print a generic MCP config block:

```bash
python -m scripts.repo_library_mcp --print-config-json
```

See [docs/agent_access.md](docs/agent_access.md) for the tool inventory and the recommended flow for Codex, Claude Code, and other MCP-capable clients.
