## Local ArXiv Metadata Snapshot

This repository assumes a **local snapshot of ArXiv metadata** is available
under `/data/arxiv`. The intent is to make it easy to search and index
papers without repeatedly hitting external APIs.

### Location and layout

- Root directory: `/data/arxiv`
- Typical contents (from the Kaggle Cornell ArXiv dataset):
  - `arxiv-metadata-oai-snapshot.json` — line-delimited JSON, one paper per line.
  - Additional CSV/JSON files may be present depending on the specific Kaggle
    bundle, but the metadata JSON snapshot is the primary artifact.

### Metadata format (JSONL)

Each line in `arxiv-metadata-oai-snapshot.json` is a JSON object describing
one ArXiv paper. Common fields include (names may vary slightly with dataset
version):

- `id`: ArXiv identifier, e.g. `"hep-th/9901001"` or `"2101.00001"`.
- `submitter`: free-text submitter name.
- `title`: paper title.
- `authors`: comma-separated author string.
- `categories`: space-separated ArXiv category tags, e.g. `"cs.CL cs.LG"`.
- `doi`: optional DOI string.
- `journal-ref`: optional journal reference.
- `abstract`: full abstract text.
- `update_date` / `versions`: versioning information.

Because this is JSONL, you should treat the file as a stream of independent
JSON objects rather than a single JSON array.

### Basic usage patterns

#### 1. Quick filtering / grep

For ad-hoc shell exploration:

- Search titles or abstracts:
  - `grep -i "transformer" /data/arxiv/arxiv-metadata-oai-snapshot.json`
- Restrict to a category:
  - `grep '"categories": "cs.CL' /data/arxiv/arxiv-metadata-oai-snapshot.json`

#### 2. Python: iterating and searching

When integrating with the repository library, prefer streaming reads to avoid
loading the entire snapshot into memory:

```python
import json
from pathlib import Path

SNAPSHOT_PATH = Path("/data/arxiv/arxiv-metadata-oai-snapshot.json")

def iter_arxiv_records():
    with SNAPSHOT_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def search_by_keyword(keyword: str, max_results: int = 50):
    kw = keyword.lower()
    out = []
    for rec in iter_arxiv_records():
        text = f"{rec.get('title','')} {rec.get('abstract','')}".lower()
        if kw in text:
            out.append(rec)
            if len(out) >= max_results:
                break
    return out
```

You can use helpers like this to:

- Build a lightweight search API over `/data/arxiv`.
- Construct embedding indices (e.g., titles + abstracts) for downstream QA.

### Integration with the QA / skills stack

For now, the ArXiv metadata is **standalone** and not wired into the
`RepoLibrary` or skill adapters. Typical next steps to integrate it would be:

- Build a small embedding index over titles/abstracts under
  `/data/arxiv/indices/`.
- Expose a simple FastAPI endpoint (e.g. `/api/arxiv/search`) that:
  - Accepts a natural language query.
  - Searches the local snapshot.
  - Returns matching paper IDs, titles, abstracts, and categories.
- Optionally, add a new skill (e.g. `"arxiv_qa"`) that:
  - Uses this index as retrieval.
  - Runs the existing LLaMA-based QA model to answer “over” the retrieved
    abstracts.

This document only covers **metadata**; full PDF content is intentionally out
of scope and is not stored under `/data/arxiv` by default.


