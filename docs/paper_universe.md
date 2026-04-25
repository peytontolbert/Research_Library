## Paper Universe Layout

### Concept

- We maintain a separate paper-centric universe for the paper corpus, analogous
  to the repository universe.
- Each paper is a node with a semantic embedding derived from paper metadata.
- The default semantic view is `title + abstract`, which keeps million-paper
  builds practical while still preserving topic-level structure.
- arXiv categories become lightweight anchor nodes, so the graph has both:
  - paper nodes
  - category nodes
  - membership edges from papers to categories
- The default rendered/viewer experience should be paper-first: sampled papers
  are the plotted nodes, while category/year layers are optional anchors for
  navigation and orientation.

This is intentionally separate from the repository universe:

- repository universe: code entities, repos, repo-to-repo similarity
- paper universe: papers, categories, optional paper-to-paper similarity

### Outputs

Running `scripts/paper_universe_build.py` writes:

- `paper_nodes.parquet`
- `category_nodes.parquet`
- `year_nodes.parquet`
- `edges.parquet`
- `paper_year_edges.parquet`
- `paper_embeddings.parquet`
- `paper_fulltext_embeddings.parquet` (optional)
- optional `paper_knn_edges.parquet`
- optional `category_knn_edges.parquet`
- `topic_nodes.parquet`
- `paper_topic_edges.parquet`
- `manifest.json`
- `progress.json`

The paper universe does not duplicate full paper text. It stores lightweight
paper references and metadata (`paper_id`, `canonical_paper_id`, version, title,
authors, categories, PDF path, counts), plus embeddings and 3D coordinates.
The original full text remains in the source paper dataset.

During the build, committed temp shards are stored under:

- `_build_tmp/paper_rows_*.parquet`
- `_build_tmp/paper_embeddings_*.parquet`
- `_build_tmp/edges_*.parquet`
- `_build_tmp/shard_*.json`

Those temp shards make restart/resume safe: if the process is interrupted, a
rerun resumes from the last committed shard instead of re-embedding from zero.

### Recommended build

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. python -m scripts.paper_universe_build \
  --dataset-dir /arxiv/huggingface/paper_text_1m_dedup_v1 \
  --output-dir /data/repository_library/exports/_paper_universe \
  --batch-rows 1024 \
  --embed-batch-size 128 \
  --embed-device cuda:0 \
  --temp-shard-rows 16384 \
  --parquet-compression zstd \
  --fulltext-max-chunks 4 \
  --fulltext-chunk-chars 2000 \
  --max-topics-per-paper 3 \
  --paper-knn 10 \
  --category-knn 8
```

When `CUDA_VISIBLE_DEVICES=1` is set, physical GPU 1 becomes logical `cuda:0`
inside the process.

That produces a paper graph with:

- one node per paper
- one node per arXiv category
- one node per publication year
- lightweight keyword/topic nodes
- paper -> category edges
- paper -> year edges
- paper -> topic edges
- paper -> paper KNN edges
- category -> category similarity edges
- a 3D coordinate for every paper
- a metadata embedding parquet split (`title + abstract`)
- an optional full-text embedding parquet split

### Progress

Watch live build state in:

- `/data/repository_library/exports/_paper_universe/progress.json`

Useful fields:

- `phase`
- `processed_papers`
- `remaining_papers`
- `temp_shards_written`
- `final_paper_nodes_written`
- `status`

### Reuse Existing Base Outputs

If `paper_nodes.parquet`, `paper_embeddings.parquet`, and `edges.parquet`
already exist for the full dataset, rerunning the builder reuses that base and
adds missing derived graph layers such as:

- `paper_knn_edges.parquet`
- `category_knn_edges.parquet`
- `year_nodes.parquet`
- `paper_year_edges.parquet`
- `paper_fulltext_embeddings.parquet`
- `topic_nodes.parquet`
- `paper_topic_edges.parquet`

without re-embedding the full paper corpus again.

### Notes

- The current repo-paper alignment exports are far too small to place the full
  1M paper corpus inside the repo universe directly.
- A standalone paper universe is therefore the right first step.
- Later, we can add a cross-universe bridge layer using:
  - paper↔repo alignment edges
  - shared embedding neighborhoods
  - category/domain anchors
