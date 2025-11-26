Repository Library
==================

A backend service and knowledge system that indexes code repositories stored in `/data/repositories`, builds a graph-centric representation of them under `/data/repository_library/exports`, and (in later phases) exposes LLM-powered retrieval and editing over this **repository library**.

The current implementation focuses on:

- Exporting a **per-repository program graph** as JSONL.
- Maintaining a **library manifest** (`exports/_manifest.json`) with lightweight per-repo metadata.
- Providing the foundations for **Repository objects** and **SkillSets** that higher-level services (Q&A, refactors, meta-learning) can use.

### Goals

- **Central library of repositories**: Treat `/data/repositories` as a single searchable "library" of code and related assets.
- **Graph-based knowledge model**: Represent repositories, files, symbols, and relationships (dependencies, ownership, references) as a program graph.
- **Repository-centric skills**: For each repository, attach a set of *skills* (adapters, tools, indices) that specialize the system for Q&A, editing, navigation, testing, etc.
- **LLM-powered workflows** (later phase): Answer questions, propose edits, and distill meta-knowledge over the entire library.

### Repository-Centric View

Conceptually, each repository is normalized into a `Repository` object:

```python
class Repository:
    repo_id: str          # e.g. "linux" or "github:python/cpython"
    root_path: Path       # on-disk mirror under /data/repositories
    metadata: dict        # tags, languages, git state, etc.
    graph: ProgramGraph   # code structure, symbols, dependencies
    index: RepoIndex      # text/code embeddings, search indices
    tools: RepoTools      # repo-scoped tools (grep, tests, build, etc.)
    skills: SkillSet      # repo-conditioned adapters / skills
```

The **export pipeline** in `build.py` and `scripts/library_repo_graph_export.py` is responsible for:

- Discovering repositories under `/data/repositories`.
- Building a `PythonRepoGraph` per repo.
- Writing JSONL exports:
  - `{repo_id}.entities.jsonl`
  - `{repo_id}.edges.jsonl`
  - `{repo_id}.artifacts.jsonl`
- Updating a shared manifest in `exports/_manifest.json` with per-repo metadata and (in later steps) skillset information.

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
  - Scans `/data/repositories` on a schedule or trigger.
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
