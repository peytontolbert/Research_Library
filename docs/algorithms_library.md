## Algorithms Library

This document defines the **Algorithms Library** as a first‑class sibling to:

- `/data/repositories` (codebases)
- `/data/arxiv` (metadata)
- `/arxiv/pdfs` (papers)

The Algorithms Library lives under:

- `/data/algorithms`

and is designed to answer questions like:

- “**Which algorithms solve this problem?**”
- “**What are the variants and tradeoffs?**”
- “**Where are the Python / C++ implementations of Dijkstra?**”
- “**Which algorithm should I pick under these constraints?**”

The **center of gravity** is *abstract algorithms and problems*, deduplicated
across repositories and papers. The same Dijkstra implementation might appear
in dozens of repos and papers, but there is **one canonical Algorithm node**
for `dijkstra`.

The intent is that this library can be:

- Traversed directly from the UI (list/search algorithms, drill into problems,
  follow links to implementations and benchmarks).
- Used as a lightweight retrieval layer for QA prompts (“find algorithms for
  single_source_shortest_path under non‑negative weights, low memory”).

This document is **source of truth** for the on‑disk layout and schemas.

---

### 1. Data layout under `/data/algorithms`

All core entities are stored as **JSONL** (one JSON object per line), similar
in spirit to the ArXiv metadata snapshot.

- Root directory:
  - `/data/algorithms`

- Core files:
  - `/data/algorithms/algorithms.jsonl`
  - `/data/algorithms/problems.jsonl`
  - `/data/algorithms/implementations.jsonl`
  - `/data/algorithms/benchmarks.jsonl`
  - `/data/algorithms/topics.jsonl` (optional / small)

Each file is *append‑only* and treated as a stream; you should not assume the
entire file fits in memory. Readers should stream over entries and filter.

#### 1.1. `algorithms.jsonl`

Each line is a single **Algorithm** node:

- **Primary key**: `algo_id` (stable, slug‑style identifier).

Recommended shape:

```json
{
  "algo_id": "dijkstra",
  "names": ["Dijkstra", "Dijkstra's algorithm"],
  "category": "shortest_path",
  "problems": ["single_source_shortest_path"],
  "time_complexity": {
    "best": "O(E + V log V)",
    "avg": "O(E + V log V)",
    "worst": "O(E + V log V)"
  },
  "space_complexity": {
    "worst": "O(V)"
  },
  "properties": {
    "exact": true,
    "online": false,
    "incremental": false,
    "deterministic": true
  },
  "constraints": {
    "edge_weights": "non_negative",
    "graph_type": "directed_or_undirected"
  },
  "notes": "Classic label-setting shortest path algorithm with a priority queue.",
  "topics": ["graphs", "greedy", "shortest_path"],
  "tags": ["introductory", "core"]
}
```

Fields:

- `algo_id` (string, required): canonical identifier (e.g. `dijkstra`,
  `bellman_ford`, `a_star`, `quicksort`).
- `names` (list[string]): human‑readable names and variants.
- `category` (string): rough family (e.g. `shortest_path`, `sorting`, `max_flow`).
- `problems` (list[string]): list of `problem_id` values this algorithm solves.
- `time_complexity` (object): free‑form structured strings for big‑O.
- `space_complexity` (object): free‑form structured strings for memory.
- `properties` (object): Booleans/strings for things like `stable`,
  `online`, `approximate`, `exact`, `incremental`, etc.
- `constraints` (object): high‑level applicability constraints
  (`edge_weights`, `graph_type`, `input_size`, etc.).
- `notes` (string): short textual description.
- `topics` (list[string]): topical tags (see `topics.jsonl`).
- `tags` (list[string]): misc labels (e.g. `introductory`, `heuristic`).

#### 1.2. `problems.jsonl`

Each line is a single **Problem** node:

- **Primary key**: `problem_id` (stable, slug‑style identifier).

```json
{
  "problem_id": "single_source_shortest_path",
  "names": ["Single-source shortest path", "SSSP"],
  "description": "Given a weighted graph G = (V, E) and a source vertex s, compute the shortest path distance from s to every other vertex.",
  "topics": ["graphs", "shortest_path"],
  "constraints": {
    "graph_type": "directed_or_undirected",
    "edge_weights": "any"
  },
  "notes": "Canonical graph problem; many algorithms specialize based on edge-weight properties or graph structure."
}
```

Fields:

- `problem_id` (string, required): canonical identifier.
- `names` (list[string]): human‑friendly names and abbreviations.
- `description` (string): problem statement.
- `topics` (list[string]): topical tags.
- `constraints` (object): high‑level assumptions about inputs.
- `notes` (string): additional commentary / links.

#### 1.3. `implementations.jsonl`

Each line is an **Implementation** node: a *thin pointer* from an Algorithm
to concrete code in a repository (or another code source).

**Primary key**: `impl_id` (string) – e.g. `dijkstra_py_repo123_fileX_funcY`.

```json
{
  "impl_id": "dijkstra_py_linux_net_algo",
  "algo_id": "dijkstra",
  "language": "python",
  "repo_id": "linux",
  "repo_root": "/data/repositories/linux",
  "file_path": "net/dijkstra.py",
  "entry_symbol": "dijkstra",
  "constraints": {
    "graph_representation": "adjacency_list",
    "edge_weights": "non_negative"
  },
  "environment": {
    "python": "3.11",
    "dependencies": ["networkx>=3.0"]
  },
  "notes": "Idiomatic Python implementation with a binary heap priority queue."
}
```

Fields:

- `impl_id` (string, required): unique identifier for this implementation.
- `algo_id` (string, required): foreign key into `algorithms.jsonl`.
- `language` (string): e.g. `python`, `cpp`, `rust`, `java`.
- `repo_id` (string): matches the Repository Library manifest (`_manifest.json`).
- `repo_root` (string, optional): absolute path to repo; usually redundant
  with the manifest but convenient when browsing data outside the server.
- `file_path` (string, optional): repo‑relative path to the file.
- `entry_symbol` (string, optional): function / class name implementing the algorithm.
- `constraints` (object): implementation‑specific constraints (graph
  representation, approximate vs exact, etc.).
- `environment` (object): language/runtime hints (version, deps).
- `notes` (string): descriptive text.

#### 1.4. `benchmarks.jsonl`

Each line is a **Benchmark** measurement for a specific implementation
on a specific dataset / scenario.

**Primary key**: `benchmark_id` (string) – e.g. `dijkstra_py_linux_net_algo_random1M`.

```json
{
  "benchmark_id": "dijkstra_py_linux_net_algo_random1M",
  "impl_id": "dijkstra_py_linux_net_algo",
  "dataset_id": "random_graph_1M",
  "dataset_description": "Random graph with 1M nodes, 5M edges, non-negative weights.",
  "metrics": {
    "runtime_ms": 1234.5,
    "memory_mb": 512.0,
    "accuracy": 1.0
  },
  "environment": {
    "hardware": "2x3090",
    "cpu": "AMD EPYC 7xx",
    "os": "Ubuntu 22.04"
  },
  "notes": "Single-threaded run with default compilation flags."
}
```

Fields:

- `benchmark_id` (string, required): unique id for this benchmark row.
- `impl_id` (string, required): foreign key into `implementations.jsonl`.
- `dataset_id` (string): identifier for dataset or synthetic config.
- `dataset_description` (string): human‑readable description.
- `metrics` (object): numeric metrics (`runtime_ms`, `memory_mb`, `accuracy`, etc.).
- `environment` (object): hardware/software environment.
- `notes` (string): commentary.

Benchmarks are **optional**; the algorithms library should be useful even with
just Algorithm + Problem + Implementation populated.

#### 1.5. `topics.jsonl` (optional)

Simple reference list of **Topic / Tag** nodes:

```json
{
  "topic_id": "graphs",
  "names": ["Graphs"],
  "description": "Graph theory and graph algorithms.",
  "kind": "domain"
}
```

Fields:

- `topic_id` (string, required): canonical identifier (e.g. `graphs`, `dp`,
  `approximation`, `network_flow`).
- `names` (list[string]): human names.
- `description` (string): description.
- `kind` (string): optional, e.g. `domain`, `technique`, `property`.

`topics` connect to Algorithms and Problems via the `topics` arrays in their
respective JSON objects (no separate edge file is required).

---

### 2. Relationships / graph view

Conceptually, the Algorithms Library forms a small graph:

- `Algorithm(algo_id)`
- `Problem(problem_id)`
- `Implementation(impl_id)`
- `Benchmark(benchmark_id)`
- `Topic(topic_id)`

With edges:

- `Algorithm` -[:SOLVES]-> `Problem`
  - Encoded as `algo.problems = [problem_id, ...]`.
- `Algorithm` -[:TAGGED_AS]-> `Topic`
  - Encoded as `algo.topics = [topic_id, ...]`.
- `Problem` -[:TAGGED_AS]-> `Topic`
  - Encoded as `problem.topics = [topic_id, ...]`.
- `Implementation` -[:IMPLEMENTS]-> `Algorithm`
  - Encoded as `impl.algo_id`.
- `Implementation` -[:LIVES_IN]-> `Repository`
  - Encoded as `impl.repo_id` (joins to the Repository Library manifest).
- `Benchmark` -[:MEASURES]-> `Implementation`
  - Encoded as `benchmark.impl_id`.

Nothing here requires a separate graph database; the JSONL files are the
authoritative source, and traversals can be implemented in‑process.

---

### 3. QA‑friendly retrieval patterns

The primary **QA flows** to support:

- **Problem → Algorithms**
  - Given a problem description or `problem_id`, list candidate algorithms that
    solve it, plus brief tradeoffs.
- **Algorithm → Implementations**
  - Given an `algo_id` (e.g. `dijkstra`) and language constraints, surface
    relevant implementations (Python, C++, Rust) across repositories.
- **Constraints → Algorithm choice**
  - Given a natural‑language question with constraints (“graph is a DAG, edges
    may be negative, I care about memory more than speed”), retrieve algorithms
    whose `constraints`, `properties`, and `topics` roughly match.

Recommended patterns:

- Maintain **lightweight keyword search** over Algorithm + Problem text fields:
  - `algorithms.names`, `algorithms.notes`, `problems.description`.
- Filter / rank by:
  - `problems` / `problem_id`
  - `topics`
  - `properties` / `constraints`
- For LLM‑backed QA:
  - Retrieve top‑k Algorithms + Problems + Implementations matching the query.
  - Construct a compact textual context block (name, problem statement,
    complexities, constraints, implementation pointers).
  - Let the QA model choose / explain a recommendation.

The initial implementation keeps this in a **pure Python module** (see
`modules/algorithms_library.py`) with simple streaming + filtering. Embedding
indices under `/data/algorithms/indices` can be added later without changing
the on‑disk JSONL format.

---

### 4. Minimal Python interface (`modules.algorithms_library`)

The canonical in‑process interface should mirror `modules.arxiv_library`:

- Constants:
  - `ALGORITHMS_ROOT = Path("/data/algorithms")`
  - `ALGORITHMS_PATH = ALGORITHMS_ROOT / "algorithms.jsonl"`
  - `PROBLEMS_PATH = ALGORITHMS_ROOT / "problems.jsonl"`
  - `IMPLEMENTATIONS_PATH = ALGORITHMS_ROOT / "implementations.jsonl"`
  - `BENCHMARKS_PATH = ALGORITHMS_ROOT / "benchmarks.jsonl"`

- Dataclasses:
  - `Algorithm`, `Problem`, `Implementation`, `Benchmark`
    - Each mirrors the JSONL schema but is tolerant of missing/extra fields.

- Iterators:
  - `iter_algorithms() -> Iterator[Algorithm]`
  - `iter_problems() -> Iterator[Problem]`
  - `iter_implementations() -> Iterator[Implementation]`
  - `iter_benchmarks() -> Iterator[Benchmark]`

- Search helpers:
  - `search_algorithms(query: str, *, problem_id: Optional[str], topic: Optional[str], max_results: int = 50) -> List[Dict[str, object]]`
    - Keyword search + filters; returns JSON‑serializable dicts for API use.

The first version deliberately **does not** build or depend on any external
vector database; it uses streaming keyword matching, mirroring the
`modules.arxiv_library.search_keyword` helper.

---

### 5. API and UI integration (first pass)

The FastAPI server (`run.py`) should expose lightweight endpoints that align
with the above module:

- `GET /api/algorithms`
  - List algorithms with optional query params:
    - `problem_id` (string, optional)
    - `topic` (string, optional)
  - Returns summaries: `algo_id`, `names`, `category`, `problems`, `topics`.

- `POST /api/algorithms/search`
  - Body:
    - `{"query": "...", "problem_id": "...?", "topic": "...?", "max_results": 50}`
  - Returns:
    - `{"type": "algorithm_search_result", "query": "...", "count": N, "results": [...]}`.

UI traversal (inline HTML in `run.py`) can then:

- Render a side panel or section for Algorithms (similar to Repositories).
- Allow free‑text search + filters by problem/topic.
- On selection, show:
  - Algorithm metadata (complexities, properties, constraints).
  - Linked problems and topics.
  - Linked implementations (with `repo_id`, `file_path`, `entry_symbol`).

The QA stack can treat these endpoints / helpers as an additional *retrieval
layer* when planning answers, but the first milestone is to have:

- A clean, documented `/data/algorithms` layout.
- A small, robust `modules.algorithms_library` module.
- Simple JSON APIs for listing/searching algorithms.
---

### 6. Programmatic ingestion beyond seeds (e.g. `python_algorithms`)

Once the initial seeds (`algorithms.jsonl`, `problems.jsonl`) are in place, the
typical next step is to **auto-populate `implementations.jsonl`** from one or
more code repositories that contain concrete algorithm implementations.

A canonical example is a local clone of
[`TheAlgorithms/Python`](https://github.com/TheAlgorithms/Python) living at:

- `/data/repository_library/python_algorithms`

This section describes a pragmatic ingestion pattern for that repo (or any
similar “algorithm zoo” repo).

#### 6.1. Treat seeds as canonical ontology

The **seed files are the ontology**:

- `/data/algorithms/algorithms.jsonl`
  - Defines stable `algo_id` values (`dijkstra`, `bellman_ford`, `quicksort`, …).
- `/data/algorithms/problems.jsonl`
  - Defines stable `problem_id` values (`single_source_shortest_path`, …).

Ingestion scripts **do not invent new `algo_id` or `problem_id` values by
default**. Instead, they:

- Map files/functions in `python_algorithms` onto existing `algo_id`s.
- Emit `Implementation` rows that point from these `algo_id`s to concrete
  Python files and symbols.

If you later extend the ontology (add new algorithms/problems), you:

1. Edit the seed JSONL (or its source in `seeds/algorithms/`).
2. Re-run ingestion to attach new implementations where relevant.

#### 6.2. Choose a `repo_id` for `python_algorithms`

For consistency with the Repository Library, pick a stable `repo_id` to
represent the `python_algorithms` directory, for example:

- `repo_id = "python_algorithms"`
- `repo_root = "/data/repository_library/python_algorithms"`

You can either:

- Add this repo to the main `_manifest.json` (under `/data/repositories`) and
  mirror its root there, or
- Treat it as a “standalone” source for the Algorithms Library only and fill
  `repo_root` directly in `implementations.jsonl`.

The **recommended** path is to eventually add it to the Repository Library
manifest so that QA/graph tooling can also reason about its code, but the
Algorithms Library does not require that upfront.

#### 6.3. Define a small mapping config (`algo_id` ↔ files)

Because directory/file names in `python_algorithms` are not always a perfect
match for `algo_id`s, use a small version-controlled mapping file to anchor
ingestion. For example, create:

- `seeds/algorithms/python_algorithms_mappings.yaml`

with content like:

```yaml
repo_id: python_algorithms
repo_root: /data/repository_library/python_algorithms

mappings:
  # Graph algorithms
  - algo_id: dijkstra
    files:
      - graphs/dijkstras.py
  - algo_id: bellman_ford
    files:
      - graphs/bellman_ford.py
  - algo_id: bfs
    files:
      - graphs/breadth_first_search.py
  - algo_id: dfs
    files:
      - graphs/depth_first_search.py

  # Sorting
  - algo_id: quicksort
    files:
      - sorts/quick_sort.py
  - algo_id: mergesort
    files:
      - sorts/merge_sort.py
  - algo_id: heapsort
    files:
      - sorts/heap_sort.py

  # Search
  - algo_id: binary_search
    files:
      - searches/binary_search.py
```

This file:

- Tells the ingestion script which `algo_id` each file should attach to.
- Can be extended incrementally as you decide which parts of
  `python_algorithms` you care about.

You can use `DIRECTORY.md` in the `python_algorithms` repo (mirroring
[`DIRECTORY.md` in TheAlgorithms/Python](https://github.com/TheAlgorithms/Python/blob/master/DIRECTORY.md))
as a reference while building up this mapping.

#### 6.4. Ingestion script: from mapping → `implementations.jsonl`

Create a small script, e.g.:

- `scripts/ingest_algorithms_from_python_algorithms.py`

Its responsibilities:

1. **Load ontology and mappings**
   - Stream `algorithms.jsonl` into a dict `algo_id -> Algorithm`.
   - Load `python_algorithms_mappings.yaml` into memory.
   - Validate: every `algo_id` referenced in the mapping exists in
     `algorithms.jsonl`.

2. **Walk files and emit Implementation rows**
   For each mapping entry:

   - For each `file_path` (e.g. `graphs/dijkstras.py`):
     - Compute a stable `impl_id`, e.g.:
       - `impl_id = f"{algo_id}_py_{repo_id}_{file_path.replace('/', '_').replace('.py', '')}"`
     - Fill an `Implementation` JSON object:

       ```json
       {
         "impl_id": "dijkstra_py_python_algorithms_graphs_dijkstras",
         "algo_id": "dijkstra",
         "language": "python",
         "repo_id": "python_algorithms",
         "repo_root": "/data/repository_library/python_algorithms",
         "file_path": "graphs/dijkstras.py",
         "entry_symbol": null,
         "constraints": {},
         "environment": {
           "python": "3.11"
         },
         "notes": "Imported from python_algorithms graphs/dijkstras.py"
       }
       ```

   - Append one JSON line per implementation to:
     - `/data/algorithms/implementations.jsonl`
       - or an intermediate file such as
         `seeds/algorithms/python_algorithms_implementations.jsonl` that you
         then concatenate into the main file.

3. **(Optional) Detect `entry_symbol` automatically**

   As an upgrade, the script can parse the Python file (using `ast` or your
   existing `PythonRepoGraph`) and:

   - Look for a top-level function or class whose name closely matches
     `algo_id` or the module name (e.g. `dijkstras` / `dijkstra`).
   - Set `entry_symbol` accordingly.

   This is not required for the Algorithms Library to be useful, but it makes
   navigation and QA prompts more precise.

4. **Deduplication / idempotence**

   When re-running the script, you can:

   - Either rewrite `implementations.jsonl` from scratch (easiest), or
   - Maintain a small index of `(algo_id, repo_id, file_path)` to avoid duplicate
     implementations when appending.

   The simplest initial pattern is:

   - Keep a `seeds/algorithms/python_algorithms_implementations.jsonl`
     generated by the script.
   - When updating, regenerate that file and then replace or rebuild the main
     `/data/algorithms/implementations.jsonl` by concatenating:
     - `python_algorithms_implementations.jsonl`
     - implementations from other sources.

#### 6.5. End-to-end workflow

Putting it together, an end-to-end update flow looks like:

1. **Edit ontology seeds (if needed)**
   - Update `seeds/algorithms/algorithms.jsonl` and/or
     `seeds/algorithms/problems.jsonl`.
   - Copy/sync them into `/data/algorithms`.

2. **Extend/review mappings**
   - Edit `seeds/algorithms/python_algorithms_mappings.yaml` to cover more
     files under `/data/repository_library/python_algorithms`.

3. **Run ingestion script**
   - Execute `scripts/ingest_algorithms_from_python_algorithms.py` (to be
     implemented following the above spec).
   - This produces or updates a JSONL file with `Implementation` records.
   - Sync that output into `/data/algorithms/implementations.jsonl`.

4. **Verify via API / UI**
   - Hit `GET /api/algorithms` or `POST /api/algorithms/search` with
     `topic="graphs"` or `problem_id="single_source_shortest_path"`.
   - Confirm that the response includes implementations pointing into the
     `python_algorithms` repo (with correct `repo_id`, `file_path`).

At this point, the Algorithms Library will:

- Know canonical `Algorithm` and `Problem` nodes (from seeds).
- Know where their Python implementations live in `python_algorithms` (from
  ingestion).
- Be traversable both via the JSON API and via the UI, and immediately
  usable as a retrieval layer for QA planning.

