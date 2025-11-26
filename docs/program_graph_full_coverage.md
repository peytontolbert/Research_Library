## Program Graph Full-Coverage Upgrade

This document describes the planned evolution of the Repository Library's
program graph from a Python-centric structure to a broader, multi-language,
file-aware view that better represents large real-world codebases such as
`linux`, `transformers`, `cpython`, `v8`, and `torch`.

The goal is **not** to perfectly model every language and build system, but
to reach a pragmatic level of coverage that:

- Treats *all* relevant files as first-class graph entities.
- Adds lightweight structure for key non-Python languages (C/C++ and JS/TS).
- Preserves backwards compatibility with the existing Python graph and
  export formats.
- Keeps Git commits as metadata for now (no full commit graph yet).

This is intended as a living document that corresponds to the high-level
plan in `todo.md` under "Program Graph Full-Coverage Upgrade".

---

### 0. Current architecture (baseline)

**Core components:**

- `modules/program_graph.py`
  - Defines the abstract `ProgramGraph` interface:
    - `entities() -> Iterable[Entity]`
    - `edges() -> Iterable[Edge]`
    - `artifacts(kind) -> Iterable[Artifact]`
    - `search_refs`, `resolve`, `subgraph`
  - Defines `Entity`, `Edge`, `Artifact`, and `ResolvedAnchor` dataclasses.

- `scripts/repo_graph.py :: RepoGraph`
  - A repo-scoped implementation of `ProgramGraph`:
    - Tracks `repo_root`, `program_id`, basic file hashing.
    - Implements `artifacts(kind="source")` over **all files** discovered
      via `_discover_files`.
    - Delegates URI resolution for entities to subclasses.

- `scripts/codegraph_core.py :: CodeGraph`
  - Python-only static analysis:
    - Walks `.py` files.
    - Builds entities for:
      - `module` (one per file).
      - `function` / `class`.
      - `test_module` (for test-like filenames).
    - Builds edges for:
      - `owns` (module → function/class).
      - `imports` (module-level imports).
      - `calls` (best-effort call targets based on identifier names).

- `scripts/python_repo_graph.py :: PythonRepoGraph`
  - Wraps `CodeGraph` into a `RepoGraph`:
    - `entities()` exposes Python entities as `ProgramGraph.Entity` nodes.
    - `edges()` exposes Python edges.
    - `artifacts(kind)` surfaces per-file `Artifact`s using `CodeGraph.file_hash`.

- `scripts/library_repo_graph_export.py`
  - Uses `PythonRepoGraph` to export:
    - `*.entities.jsonl`
    - `*.edges.jsonl`
    - `*.artifacts.jsonl`
  - Maintains `_manifest.json` with:
    - `repo_state` (Git info).
    - Per-repo `languages` list derived from file entities' language tags.
    - Indices/skills metadata.

- `modules/vector_index.py :: build_repo_qa_index`
  - Enumerates `repo.graph.entities()` and builds a QA index over those
    entities only (no non-Python structure).

**Limitations:**

- Python-centric: only `.py` files get structured entities/edges.
- Non-Python files appear only as `Artifact`s (file-level, no entities).
- Commits are stored as manifest metadata (`repo_state`) and exposed via
  APIs, but are not represented as graph entities.

---

### 1. Design goals for the upgrade

- **Broader file coverage**
  - Every source file of interest (C/C++/JS/TS/Python/Markdown) should map
    to at least one `Entity(kind="file")`.

- **Lightweight multi-language structure**
  - For C/C++ and JS/TS:
    - Identify top-level functions and (optionally) types/classes.
    - Attach simple `owns` edges from file → function/type entities.

- **Backwards compatibility**
  - Do not break existing:
    - Python entity IDs or URIs.
    - Export file formats.
    - QA index consumers.
  - All changes should be additive or strictly superset behavior.

- **Performance and scalability**
  - Avoid excessive work on giant repos (e.g., linux) by:
    - Keeping analyzers shallow and robust.
    - Allowing caps on entities per repo if needed.

---

### 2. File-level entities

**New entity kind: `file`**

- For every file discovered under a repo (subject to ignore rules), we will
  create a `ProgramGraph.Entity` with:

  - `kind = "file"`
  - `id = f"file:{rel_path}"` where `rel_path` is the repo-relative Unix-style path.
  - `uri = f"program://{program_id}/file/{rel_path}"`
  - `name = rel_path` (or just the basename for display).
  - `labels` may include:
    - Language tags (e.g., `lang:c`, `lang:python`, `lang:js`).
    - Other coarse categories in the future (e.g., `doc`, `config`).

**Integration strategy (current implementation):**

- `RepoGraph` provides a utility based on `_discover_files` that:
  - Enumerates all files under the repo.
  - Provides coarse language labels via a helper that maps extensions to tags
    like `lang:python`, `lang:c`, `lang:cpp`, `lang:js`, `lang:ts`, `lang:markdown`.

- `PythonRepoGraph.entities()`:
  - Preserves existing Python entities (modules, classes, functions, tests).
  - Adds `file` entities for **all** files (including `.py`) using the scheme
    described above, with appropriate language labels.
  - This ensures multi-language repos (e.g., `cpython` with `.py` and `.c`
    files) have whole-repo coverage at the file level.

**Exports and consumers:**

- `library_repo_graph_export` will automatically include these `file` entities
  since it simply walks `graph.entities()`.
- Downstream tools can:
  - Filter entities by `kind`.
  - Use `labels` to limit to specific languages.
  - Read the per-repo `languages` list recorded in the export manifest, derived
    from language tags on `file` entities.

---

### 3. Language tagging for files

To distinguish languages in a lightweight way, we will attach tags to
`file` entities based on filename extension.

**Extension → language mapping (initial draft):**

- C / C++:
  - `.c`, `.h`, `.cc`, `.cpp`, `.cxx`, `.hpp` → `lang:c`, `lang:cpp`
    (may use more precise split later).
- JavaScript / TypeScript:
  - `.js`, `.jsx`, `.mjs` → `lang:js`
  - `.ts`, `.tsx` → `lang:ts`
- Python:
  - `.py` → `lang:python`
- Markdown:
  - `.md` → `lang:markdown`
- Fallback:
  - Files with unknown or unsupported extensions may omit language tags
    or get a generic `lang:unknown`.

These tags will live in `Entity.labels` so that:

- QA index builders can choose which languages to include.
- UI or analytics tools can color/cluster files by language.

---

### 4. Basic structure for C/C++ and JS/TS

For a first pass, the goal is **not** deep semantic understanding but:

- Identifying top-level functions and, when cheap, types/classes.
- Expressing ownership edges from files to those symbols.

**Approach options:**

- **Minimal, regex/heuristic-based:**
  - Pros: No new heavy dependencies; simpler to ship.
  - Cons: Less precise, especially for complex C++ and JS/TS code.

- **Tree-sitter-based parsers (recommended for quality, if available):**
  - Use Tree-sitter grammars for C, C++, JavaScript, and TypeScript.
  - Build thin wrappers that:
    - Parse a file.
    - Walk the syntax tree to identify function and type declarations.
    - Return `Entity` + `Edge` records.

**Entity design for non-Python symbols:**

- `kind`:
  - Could reuse `function` / `class`, plus language tags in `labels`, or
  - Use language-specific kinds like `c_function`, `js_function`.
- `id`:
  - Should be stable and unique inside the repo.
  - Could follow a pattern like:
    - `c:{rel_path}::{name}`
    - `js:{rel_path}::{name}`
- `uri`:
  - Use the same `program://` scheme but with a language-appropriate `kind`
    or resource path.

**Edges:**

- At minimum:
  - `owns` edges from file entities to the function/type entities they contain.
- Optionally (later):
  - `calls` edges for obvious direct calls.
  - `imports`-like relationships for includes or module imports.

**Current Python implementation:**

- `CodeGraph` continues to model:
  - `owns` edges from Python modules to classes/functions.
  - `imports` and `calls` edges at the module level.
- `PythonRepoGraph` augments this with:
  - `owns` edges from `file` entities for `.py` files to their corresponding
    Python module entities.
  - This yields a two-level hierarchy for Python:
    - `file` → `module` → `class`/`function`.

**Integration into the graph:**

- Either:
  - Extend `PythonRepoGraph` to also call C/C++/JS/TS analyzers, or
  - Introduce a new `MultiLangRepoGraph` that composes:
    - `CodeGraph` (Python).
    - `C/C++` analyzer.
    - `JS/TS` analyzer.

In either case, `entities()` and `edges()` will be the merged view.

---

### 5. QA index considerations

Currently, `build_repo_qa_index` uses **graph entities** and represents each
entity as a short text like `"<kind> <name>"`.

With the upgraded graph:

- There are new entity kinds:
  - `file` for all files (with language labels).
  - Future non-Python function/type entities.
- The current implementation of `build_repo_qa_index`:
  - Enumerates all entities from the repo’s `ProgramGraph`.
  - Filters to a subset of **indexable** entities:
    - Always includes structured code entities:
      - `kind in {"module","class","function"}`.
    - Includes `file` entities only when:
      - `kind == "file"`, and
      - `labels` contain a known text/source language tag such as
        `lang:python`, `lang:c`, `lang:cpp`, `lang:js`, `lang:ts`,
        or `lang:markdown`.
  - Caps the number of entities by `max_entities` (default: 20k).
- The QA index format does not need to change — only the mix of entities
  fed into it has expanded in a controlled way.

---

### 6. Commits (out of scope for ProgramGraph for now)

Commits are currently:

- Tracked in the manifest (`repo_state` with `head` and `branch`).
- Exposed via APIs (e.g., commit count in `/api/repos/{repo_id}`).

For this upgrade:

- We will **not** introduce `Entity(kind="commit")` or commit edges into the
  `ProgramGraph`.
- Commit-aware features remain:
  - Manifest-level metadata.
  - UI-level displays and analytics.

A future phase could add a sampled commit graph (e.g., last N commits) if
needed, but that is explicitly out-of-scope for this iteration.

---

### 7. Rollout and validation

**Incremental rollout:**

- Implement file entities and language tags first.
- Validate on a small set of repos (including one large C-heavy repo such as
  `linux` or `cpython`) by inspecting:
  - Exported `entities.jsonl` and `artifacts.jsonl`.
  - QA index contents.

- Use the dry-run helper `scripts/test_program_graph_upgrade.py` to:
  - Build in-memory graphs for a subset of repositories (no manifest writes).
  - Inspect per-repo summaries (entity kind counts, inferred languages).
  - Quickly sanity-check behavior on large, mixed-language repos.

- Then add basic C/C++/JS/TS structure:
  - Start with one or two representative repos per language.
  - Ensure performance is acceptable on large codebases.

**Non-goals:**

- Deep, compiler-grade semantic modeling.
- Full Git commit graph integration.
- Rust or other additional languages (at least for this phase).

As phases complete, this document should be updated to reflect:

- Which languages are supported at which level (file-only vs. structural).
- Any new entity kinds or labels.
- Observed limitations or follow-up work.


