# Distributed Paper-Text Backfill

This document covers the operational workflow for growing the paper-text corpus
with a central coordinator plus many temporary workers.

The intended model is:

- the central machine owns the parquet output under `/arxiv`
- the coordinator hands out work over HTTP with a bearer token
- local coordinator workers can contribute immediately
- remote workers can join and leave freely
- worker machines only keep temporary PDFs

## Files and directories

Recommended central paths:

- metadata snapshot: `/data/arxiv/arxiv-metadata-oai-snapshot.json`
- base dataset: `/arxiv/huggingface/paper_text_124k_dedup_v1`
- growth parquet shards: `/arxiv/pdfs_structured_gcs_growth_v1`
- coordinator temp PDFs: `/arxiv/tmp/repo_library_coordinator_tmp`

Recommended worker temp path:

- `/tmp/repo_library_pdfs`

Why use `/arxiv/tmp/...` for the coordinator:

- `/arxiv` is the large data volume
- `/tmp` lives on the root filesystem
- the coordinator only needs temporary storage, but it should not compete with
  the root volume if the host is already tight on `/`

## 1. Generate a new backfill token

Generate one token on the central machine and reuse it for the entire run:

```bash
export REPO_BACKFILL_TOKEN="$(
python - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
)"

echo "$REPO_BACKFILL_TOKEN"
```

Keep that token private. Every worker for the same run uses the same token.

## 2. Start a new coordinator

Run the coordinator from the repository root on the central machine:

```bash
mkdir -p /arxiv/tmp/repo_library_coordinator_tmp

PYTHONPATH=. python -m scripts.distributed_paper_text_backfill coordinator \
  --auth-token "$REPO_BACKFILL_TOKEN" \
  --bind-host 0.0.0.0 \
  --port 8787 \
  --existing-parquet-dir /arxiv/huggingface/paper_text_124k_dedup_v1 \
  --metadata-path /data/arxiv/arxiv-metadata-oai-snapshot.json \
  --out-dir /arxiv/pdfs_structured_gcs_growth_v1 \
  --temp-pdf-dir /arxiv/tmp/repo_library_coordinator_tmp \
  --target-total-papers 1000000 \
  --min-year 1980 \
  --shard-size 100000 \
  --row-group-rows 256 \
  --parquet-compression zstd \
  --lease-timeout-seconds 3600 \
  --local-workers 2 \
  --local-max-records-per-lease 16 \
  --local-extract-workers 8
```

What this does:

- resumes from the existing growth shard directory if it already exists
- starts the HTTP coordinator on port `8787`
- runs `2` local workers on the central machine immediately
- writes all returned rows into central `paper_text_backfill_*.parquet` shards

If you are restarting an existing run, reuse the same `--out-dir`.

## 3. Define the worker connection target

Each worker needs:

- the same `REPO_BACKFILL_TOKEN`
- the public or tunneled coordinator URL

Example:

```bash
export REPO_BACKFILL_TOKEN="paste-the-token-here"
export CENTRAL_URL="http://YOURIP:8787"
```

## 4. Connect one worker

The minimal worker command is:

```bash
PYTHONPATH=. python -m scripts.distributed_paper_text_backfill worker \
  --coordinator-url "$CENTRAL_URL" \
  --auth-token "$REPO_BACKFILL_TOKEN" \
  --worker-id "runpod-worker-01" \
  --temp-pdf-dir /tmp/repo_library_pdfs \
  --max-records-per-lease 16 \
  --extract-workers 8 \
  --heartbeat-interval-seconds 60
```

Requirements on a worker host:

- `poppler-utils` installed for `pdftotext`
- Python environment with the worker script dependencies
- `gsutil` available on `PATH`
- enough temp disk for the current lease batch

Each worker must use a unique `--worker-id`.

## 5. Recommended persistent worker loop

For temporary RunPod-style machines, use a restart loop so a transient reset
does not leave the pod idle:

```bash
mkdir -p /tmp/repo_library_pdfs /workspace/repository_library/logs

nohup bash -lc '
export PATH=/workspace/repoenv/bin:/usr/local/bin:/usr/bin:/bin
while true; do
  PYTHONPATH=/workspace/repository_library \
  /workspace/repoenv/bin/python -m scripts.distributed_paper_text_backfill worker \
    --coordinator-url "$CENTRAL_URL" \
    --auth-token "$REPO_BACKFILL_TOKEN" \
    --worker-id "runpod-worker-01" \
    --temp-pdf-dir /tmp/repo_library_pdfs/worker_01 \
    --max-records-per-lease 16 \
    --extract-workers 8 \
    --heartbeat-interval-seconds 60
  sleep 5
done
' >/workspace/repository_library/logs/runpod-worker-01.log 2>&1 &
```

Repeat that with a different `--worker-id` and temp subdirectory per worker.

On a `32` vCPU worker machine, a practical layout is often:

- `4` worker processes
- `--extract-workers 8` on each worker

## 6. Track progress

The coordinator writes the live progress file here:

- `/arxiv/pdfs_structured_gcs_growth_v1/distributed_backfill_progress.json`

Useful quick check:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("/arxiv/pdfs_structured_gcs_growth_v1/distributed_backfill_progress.json")
obj = json.loads(p.read_text())
print({
    "status": obj.get("status"),
    "target_total_papers": obj.get("target_total_papers"),
    "covered_ids_after": obj.get("covered_ids_after"),
    "extracted_rows": obj.get("extracted_rows"),
    "active_leases": obj.get("active_leases"),
    "direct_pdf_fallback_uses": obj.get("direct_pdf_fallback_uses"),
    "missing_downloads": obj.get("missing_downloads"),
})
PY
```

What matters most:

- `covered_ids_after`: total deduped papers covered by base plus growth
- `active_leases`: how many worker leases are currently live
- `direct_pdf_fallback_uses`: how often the downloader fell back from GCS to
  direct arXiv PDF
- `missing_downloads`: how many candidate downloads missed

## 7. Restart behavior

The coordinator persists a safe metadata resume cursor in the progress file.
That means a restarted coordinator should resume from the saved metadata offset
instead of rescanning from byte `0` every time.

A restart should reuse:

- the same `--out-dir`
- the same metadata snapshot
- the same target parameters for the same run

Workers can reconnect freely after a coordinator restart as long as they use:

- the current `REPO_BACKFILL_TOKEN`
- the current `CENTRAL_URL`

## 8. Network and auth notes

The worker protocol uses bearer-token auth, but the built-in coordinator server
does not provide TLS on its own.

Safer deployment options:

- expose it only on a trusted network
- use Tailscale or WireGuard
- or tunnel it through SSH

## 9. After the run

When the growth shards are complete, merge them into a new Hugging Face dataset
directory with:

```bash
PYTHONPATH=. python -m scripts.merge_paper_text_parquets \
  --base-parquet-dir /arxiv/huggingface/paper_text_124k_dedup_v1 \
  --backfill-parquet-dir /arxiv/pdfs_structured_gcs_growth_v1 \
  --metadata-path /data/arxiv/arxiv-metadata-oai-snapshot.json \
  --output-dir /arxiv/huggingface/paper_text_1m_dedup_v1 \
  --target-rows 1000000 \
  --rows-per-output-file 100000 \
  --compression zstd \
  --memory-limit 16GB
```

That merge step creates the upload-ready dataset directory with:

- `train_*.parquet`
- `README.md`
- `stats.json`
