Architecture: Repository Library
================================

This document describes the evolving architecture of the repository library system that:
- Indexes repositories under `/data/repositories` and any configured extension roots.
- Stores relationships in a graph database.
- Provides LLM-based retrieval and Q&A using `meta-llama/Llama-3.1-8B-Instruct` from `/data/checkpoints`.

### 1. Core Concepts & Domain Model

- **Repository**
  - Physical path under a configured library root (usually a Git repo).
  - Metadata: name, description, primary language(s), size, last indexed commit, tags.

- **File**
  - Belongs to a single `Repository`.
  - Types: source code, config, documentation, tests, etc.
  - Links to one or more `Symbol` nodes.

- **Symbol**
  - Logical code entities: classes, functions, methods, modules, endpoints, etc.
  - Has attributes like name, language, file path, signature, docstring.
  - Connected via semantic edges (e.g., `CALLS`, `IMPORTS`, `IMPLEMENTS`).

- **Topic / Tag**
  - Higher-level concepts (e.g., "payments", "auth", "kubernetes").
  - May be inferred from code, docs, or manual annotations.

### 2. Graph Data Model (First Draft)

- **Nodes**
  - `Repository(id, name, root_path, default_branch, languages, tags, last_indexed_at)`
  - `File(id, path, file_type, language, loc, repo_id)`
  - `Symbol(id, name, kind, language, signature, visibility, file_id)`
  - `Topic(id, name, description)`

- **Edges**
  - `Repository` -[:CONTAINS]-> `File`
  - `File` -[:DECLARES]-> `Symbol`
  - `Symbol` -[:CALLS]-> `Symbol`
  - `Symbol` -[:IMPORTS]-> `Symbol`
  - `Repository` -[:TAGGED_AS]-> `Topic`
  - `File` -[:TAGGED_AS]-> `Topic`
  - `Symbol` -[:TAGGED_AS]-> `Topic`

This initial schema should stay relatively small and pragmatic; it can be extended later (e.g., `Service`, `Owner`, `Team`, `Endpoint`).

### 3. Ingestion & Indexing Pipeline

**Responsibilities**

- Discover repositories under `/data/repositories` plus configured extension roots.
- For each repository:
  - Extract metadata (name, languages, git origin, branches).
  - Enumerate files; store file-level metadata in the graph.
  - Optionally parse code into symbols and relationships.
  - Generate text chunks for embedding.

**High-Level Flow**

1. **Discovery**
   - Scan the default library root plus any configured extension roots for directories containing `.git` or other repository markers.
   - Maintain an ingestion state store (e.g., last indexed commit or timestamp).

2. **Metadata Extraction**
   - For each repo, collect:
     - Basic info (name, path, main language).
     - Git information (origin URL, default branch, latest commit).

3. **File & Symbol Extraction**
   - Enumerate files with filters (ignore large binaries, vendor dirs, etc.).
   - For now, limit symbol extraction to a small set of languages (e.g., Python, TypeScript) using language-specific parsers.
   - Create/Update graph nodes and edges.

4. **Chunking & Embeddings**
   - From READMEs, docs, and code:
     - Split into semantically meaningful chunks (e.g., 150–400 tokens).
     - Associate each chunk with a graph node reference (`Repository`, `File`, `Symbol`).
   - Compute embeddings using a selected embedding model (not necessarily `meta-llama/Llama-3.1-8B-Instruct`).
   - Persist embeddings in a vector store, keyed by `(chunk_id, node_type, node_id)`.

5. **Scheduling**
   - Run ingestion periodically (e.g., cron-like) or triggered manually / via webhook.
   - Support incremental re-ingest based on git changes to avoid reprocessing everything.

### 4. LLM Retrieval & Q&A (RAG)

**Key Idea**: The LLM (`meta-llama/Llama-3.1-8B-Instruct`) never answers from scratch: it uses retrieved context (graph + text chunks) to ground responses.

- **Step 1: Query Understanding**
  - Normalize question (language, formatting).
  - Optionally classify intent:
    - "Code location / usage"
    - "High-level architecture explanation"
    - "Dependency/impact analysis"
    - "Search by topic/feature"

- **Step 2: Graph + Vector Retrieval**
  - Use an embedding model to embed the question.
  - Query vector index for top-k chunks across repositories.
  - Optionally apply:
    - Filters (repo name, language, tag).
    - Pre-filtering via graph (e.g., only symbols under a specific repo or topic).

- **Step 3: Context Assembly**
  - Deduplicate and rank retrieved chunks.
  - Group by repository or topic when possible.
  - Construct a prompt for `meta-llama/Llama-3.1-8B-Instruct`:
    - System instructions (style, safety).
    - Brief summary of user question.
    - Top-N chunks (truncated to context window).

- **Step 4: Answer Generation**
  - Call `meta-llama/Llama-3.1-8B-Instruct` via the chosen runtime, loading checkpoints from `/data/checkpoints`.
  - Post-process output:
    - Optionally add citations (file paths, line ranges).
    - Truncate or format as markdown.

### 5. API Design (Draft)

Assuming an HTTP server (REST or GraphQL) running inside the project:

- **Repository & File APIs**
  - `GET /repos`
  - `GET /repos/{id}`
  - `GET /repos/{id}/files`
  - `GET /files/{id}`

- **Graph-Oriented APIs**
  - `GET /symbols/{id}`
  - `GET /symbols/{id}/calls`
  - `GET /symbols/{id}/dependents`
  - `POST /graph/query` — accept a small query language (later phase).

- **Search & Q&A**
  - `POST /search` — semantic search over text/code.
  - `POST /query` — full RAG Q&A; body examples:
    ```json
    {
      "question": "Where is the payment authorization logic implemented?",
      "filters": {
        "repo": "payments-service",
        "language": "python"
      }
    }
    ```

### 6. Deployment & Data Layout

- **Data Directories**
  - `/data/repositories` — source of truth for codebases.
  - `/data/checkpoints` — `meta-llama/Llama-3.1-8B-Instruct` model checkpoints.
  - `/data/graph` — graph database data directory (volume).
  - `/data/vector_index` — vector index / embeddings store.

- **Runtime Components**
  - `ingestion-worker` — batch job or service for scanning & indexing.
  - `api-server` — serves HTTP routes for graph/search/Q&A.
  - `llm-service` — process that wraps `meta-llama/Llama-3.1-8B-Instruct` runtime for inference.

### 7. Repository Objects & SkillSets (Planned)

At the architecture level, we treat each repository as a structured object that higher-level systems can load from the file-based exports:

- **Repository object (conceptual)**
  - `repo_id`: library-wide identifier (e.g. `linux`, `github:python/cpython`).
  - `root_path`: absolute path under `/data/repositories`.
  - `metadata`: languages, tags, git state, etc. (seeded from `_manifest.json`).
  - `graph`: a `ProgramGraph` instance built from exports (or rebuilt on demand).
  - `index`: a `RepoIndex` instance (embeddings, search indices) built from exports.
  - `tools`: a `RepoTools` handle for repo-scoped actions (grep, tests, build, etc.).
  - `skills`: a `SkillSet` grouping repo-aware adapters (QA, edit, meta, etc.).

- **SkillSet (per-repo skills)**
  - Captures specialized capabilities per repository:
    - `qa` — Q&A about this repo.
    - `edit` — style- and invariant-aware edits.
    - `meta` — pattern extraction and summarization.
    - `nav`, `test`, `perf`, `security`, `api`, `style`, etc.
  - Implemented as adapters (weights/config) plus callables that use `graph`, `index`, and `tools`.
  - Skill metadata and artifacts (paths, versions) are intended to live alongside the graph exports and be referenced from `_manifest.json`.

Library-level code (planned) will provide:

- A small Python API to **open a repo by `repo_id`** and get back a `Repository` object.
- Utilities to **list available skills and indices** per repo using the manifest.
- Hooks for **training/registration scripts** to attach new skills and indices to existing repos.

### 8. Milestones & Open Questions

**Milestone 1: Skeleton System**
- Basic ingestion of repos → `Repository` and `File` nodes.
- Simple vector index over README/docs.
- `POST /query` endpoint wired to `meta-llama/Llama-3.1-8B-Instruct` with RAG.

**Milestone 2: Rich Graph**
- Code parsing into `Symbol` nodes and relations.
- Graph-backed search (by symbol, dependency).

**Milestone 3: Developer Experience & UI**
- CLI or minimal web UI for exploring the library.
- Simple visualizations of repo graphs.

**Open Questions**
- Final choice of:
  - Graph DB (Neo4j vs alternatives).
  - Vector DB (Qdrant/Weaviate/FAISS/etc.).
  - Embedding model and runtime.
- How to handle very large monorepos (chunking strategies, incremental indexing).
- Multi-user access control and security model.

This document should evolve with implementation; treat it as a living design reference.

