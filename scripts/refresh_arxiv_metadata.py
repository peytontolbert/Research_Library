from __future__ import annotations

"""
Incrementally refresh the local arXiv metadata snapshot from arXiv's OAI-PMH.

This script keeps the existing JSONL snapshot shape used across the repository
but updates it from the official arXiv OAI endpoint instead of relying on a
manually dropped Kaggle bundle.

Usage examples:

    python -m scripts.refresh_arxiv_metadata
    python -m scripts.refresh_arxiv_metadata --dry-run --max-records 50
    python -m scripts.refresh_arxiv_metadata --from-date 2026-04-01
    python -m scripts.refresh_arxiv_metadata --full
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence


ARXIV_OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
ARXIV_ROOT = Path("/data/arxiv")
DEFAULT_SNAPSHOT_PATH = ARXIV_ROOT / "arxiv-metadata-oai-snapshot.json"
DEFAULT_STATE_PATH = ARXIV_ROOT / "arxiv-metadata-oai-state.json"
DEFAULT_OVERLAP_DAYS = 7
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_USER_AGENT = "repository-library-arxiv-sync/1.0"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}


@dataclass
class HarvestBatch:
    response_date: str
    records: Dict[str, Dict[str, object]]
    deleted_ids: set[str]
    resumption_token: Optional[str]


@dataclass
class HarvestResult:
    response_date: str
    updates_db_path: Path
    upsert_count: int
    deleted_count: int
    requests_made: int
    max_update_date: str


@dataclass
class MergeStats:
    replaced_existing: int = 0
    deleted_existing: int = 0
    appended_new: int = 0
    kept_existing: int = 0


def _normalize_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _itertext_normalized(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return _normalize_text("".join(elem.itertext()))


def _find_text(elem: ET.Element, path: str) -> str:
    return _itertext_normalized(elem.find(path, NS))


def _parse_iso_date(value: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_version_created_date(value: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%a, %d %b %Y %H:%M:%S GMT").date()
    except ValueError:
        return None


def _snapshot_latest_date(snapshot_path: Path) -> Optional[date]:
    latest: Optional[date] = None
    if not snapshot_path.is_file():
        return None

    with snapshot_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            candidate = _parse_iso_date(str(obj.get("update_date") or ""))
            if candidate is None:
                versions = obj.get("versions")
                if isinstance(versions, list):
                    for version in reversed(versions):
                        if not isinstance(version, dict):
                            continue
                        candidate = _parse_version_created_date(str(version.get("created") or ""))
                        if candidate is not None:
                            break
            if candidate is not None and (latest is None or candidate > latest):
                latest = candidate

    return latest


def _load_state_date(state_path: Path) -> Optional[date]:
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_iso_date(str(payload.get("last_response_date") or ""))


def infer_from_date(snapshot_path: Path, state_path: Path, overlap_days: int) -> Optional[str]:
    state_date = _load_state_date(state_path)
    if state_date is not None:
        return state_date.isoformat()

    snapshot_date = _snapshot_latest_date(snapshot_path)
    if snapshot_date is None:
        return None
    rewind = max(0, int(overlap_days))
    return (snapshot_date - timedelta(days=rewind)).isoformat()


def _arxiv_id_from_oai_identifier(identifier: str) -> str:
    prefix = "oai:arXiv.org:"
    text = str(identifier or "").strip()
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _parse_raw_record(header: ET.Element, raw: ET.Element) -> Dict[str, object]:
    versions = []
    for version in raw.findall("raw:version", NS):
        entry: Dict[str, object] = {}
        version_label = _normalize_text(version.get("version"))
        if version_label:
            entry["version"] = version_label
        created = _find_text(version, "raw:date")
        if created:
            entry["created"] = created
        size = _find_text(version, "raw:size")
        if size:
            entry["size"] = size
        source_type = _find_text(version, "raw:source_type")
        if source_type:
            entry["source_type"] = source_type
        if entry:
            versions.append(entry)

    obj: Dict[str, object] = {
        "id": _find_text(raw, "raw:id"),
        "submitter": _find_text(raw, "raw:submitter"),
        "authors": _find_text(raw, "raw:authors"),
        "title": _find_text(raw, "raw:title"),
        "comments": _find_text(raw, "raw:comments"),
        "journal-ref": _find_text(raw, "raw:journal-ref"),
        "doi": _find_text(raw, "raw:doi"),
        "report-no": _find_text(raw, "raw:report-no"),
        "categories": _find_text(raw, "raw:categories"),
        "license": _find_text(raw, "raw:license"),
        "abstract": _find_text(raw, "raw:abstract"),
        "versions": versions,
        "update_date": _find_text(header, "oai:datestamp"),
    }
    return obj


def parse_oai_list_records(xml_text: str) -> HarvestBatch:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"failed to parse OAI response: {exc}") from exc

    error = root.find("oai:error", NS)
    if error is not None:
        code = _normalize_text(error.get("code"))
        detail = _itertext_normalized(error)
        if code == "noRecordsMatch":
            return HarvestBatch(
                response_date=_find_text(root, "oai:responseDate"),
                records={},
                deleted_ids=set(),
                resumption_token=None,
            )
        raise RuntimeError(f"arXiv OAI returned {code or 'error'}: {detail}")

    response_date = _find_text(root, "oai:responseDate")
    token = _itertext_normalized(root.find(".//oai:resumptionToken", NS)) or None

    records: Dict[str, Dict[str, object]] = {}
    deleted_ids: set[str] = set()

    for record in root.findall(".//oai:record", NS):
        header = record.find("oai:header", NS)
        if header is None:
            continue
        paper_id = _arxiv_id_from_oai_identifier(_find_text(header, "oai:identifier"))
        if not paper_id:
            continue
        if _normalize_text(header.get("status")) == "deleted":
            deleted_ids.add(paper_id)
            records.pop(paper_id, None)
            continue

        raw = record.find("oai:metadata/raw:arXivRaw", NS)
        if raw is None:
            continue
        parsed = _parse_raw_record(header, raw)
        if parsed.get("id"):
            records[paper_id] = parsed

    return HarvestBatch(
        response_date=response_date,
        records=records,
        deleted_ids=deleted_ids,
        resumption_token=token,
    )


def _fetch_oai_page(
    *,
    base_url: str,
    metadata_prefix: str,
    from_date: Optional[str],
    set_spec: Optional[str],
    resumption_token: Optional[str],
    timeout_seconds: int,
    user_agent: str,
) -> HarvestBatch:
    if resumption_token:
        params = {"verb": "ListRecords", "resumptionToken": resumption_token}
    else:
        params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
        if from_date:
            params["from"] = from_date
        if set_spec:
            params["set"] = set_spec

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        payload = resp.read().decode("utf-8")
    return parse_oai_list_records(payload)


def harvest_records(
    *,
    base_url: str,
    metadata_prefix: str,
    from_date: Optional[str],
    set_spec: Optional[str],
    timeout_seconds: int,
    user_agent: str,
    max_records: int,
    sleep_seconds: float,
) -> HarvestResult:
    fd, db_name = tempfile.mkstemp(prefix="arxiv-oai-updates.", suffix=".sqlite3")
    os.close(fd)
    db_path = Path(db_name)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS updates (
            id TEXT PRIMARY KEY,
            payload TEXT,
            deleted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()

    response_date = ""
    max_update_date = ""
    requests_made = 0
    token: Optional[str] = None

    try:
        while True:
            batch = _fetch_oai_page(
                base_url=base_url,
                metadata_prefix=metadata_prefix,
                from_date=from_date,
                set_spec=set_spec,
                resumption_token=token,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            )
            requests_made += 1
            response_date = batch.response_date or response_date

            for deleted_id in batch.deleted_ids:
                conn.execute(
                    """
                    INSERT INTO updates(id, payload, deleted)
                    VALUES (?, NULL, 1)
                    ON CONFLICT(id) DO UPDATE SET
                        payload = NULL,
                        deleted = 1
                    """,
                    (deleted_id,),
                )

            for paper_id, obj in batch.records.items():
                payload = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
                conn.execute(
                    """
                    INSERT INTO updates(id, payload, deleted)
                    VALUES (?, ?, 0)
                    ON CONFLICT(id) DO UPDATE SET
                        payload = excluded.payload,
                        deleted = 0
                    """,
                    (paper_id, payload),
                )
                upd = str(obj.get("update_date") or "")
                if upd > max_update_date:
                    max_update_date = upd

            conn.commit()
            current_upserts = -1
            if requests_made == 1 or requests_made % 10 == 0 or not batch.resumption_token:
                current_upserts = int(
                    conn.execute("SELECT COUNT(*) FROM updates WHERE deleted = 0").fetchone()[0]
                )
                current_deleted = int(
                    conn.execute("SELECT COUNT(*) FROM updates WHERE deleted = 1").fetchone()[0]
                )
                print(
                    "Harvest progress: "
                    f"requests={requests_made}, "
                    f"upserts={current_upserts}, "
                    f"deletions={current_deleted}, "
                    f"max_update_date={max_update_date or '(missing)'}",
                    file=sys.stderr,
                )

            if max_records > 0:
                if current_upserts < 0:
                    current_upserts = int(
                        conn.execute("SELECT COUNT(*) FROM updates WHERE deleted = 0").fetchone()[0]
                    )
                if current_upserts >= max_records:
                    break

            token = batch.resumption_token
            if not token:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        upsert_count = int(conn.execute("SELECT COUNT(*) FROM updates WHERE deleted = 0").fetchone()[0])
        deleted_count = int(conn.execute("SELECT COUNT(*) FROM updates WHERE deleted = 1").fetchone()[0])
        return HarvestResult(
            response_date=response_date,
            updates_db_path=db_path,
            upsert_count=upsert_count,
            deleted_count=deleted_count,
            requests_made=requests_made,
            max_update_date=max_update_date,
        )
    except Exception:
        conn.close()
        if db_path.exists():
            db_path.unlink()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _json_dumps_line(obj: Dict[str, object]) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n"


def _target_file_mode(path: Path, *, default_mode: int = 0o664) -> int:
    if path.exists():
        return path.stat().st_mode & 0o777
    return default_mode


def merge_snapshot(
    snapshot_path: Path,
    harvested: Dict[str, Dict[str, object]],
    deleted_ids: set[str],
) -> MergeStats:
    stats = MergeStats()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    target_mode = _target_file_mode(snapshot_path)

    pending = dict(harvested)
    fd, temp_name = tempfile.mkstemp(
        prefix=snapshot_path.name + ".",
        suffix=".tmp",
        dir=str(snapshot_path.parent),
        text=True,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with temp_path.open("w", encoding="utf-8") as out_fh:
            if snapshot_path.is_file():
                with snapshot_path.open("r", encoding="utf-8") as in_fh:
                    for line in in_fh:
                        raw_line = line if line.endswith("\n") else line + "\n"
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            obj_any = json.loads(stripped)
                        except Exception:
                            out_fh.write(raw_line)
                            continue
                        if not isinstance(obj_any, dict):
                            out_fh.write(raw_line)
                            continue
                        paper_id = _normalize_text(obj_any.get("id"))
                        if not paper_id:
                            out_fh.write(raw_line)
                            continue
                        if paper_id in deleted_ids:
                            pending.pop(paper_id, None)
                            stats.deleted_existing += 1
                            continue
                        update = pending.pop(paper_id, None)
                        if update is None:
                            out_fh.write(raw_line)
                            stats.kept_existing += 1
                            continue

                        merged = dict(obj_any)
                        merged.update(update)
                        out_fh.write(_json_dumps_line(merged))
                        stats.replaced_existing += 1

            for paper_id in sorted(pending):
                if paper_id in deleted_ids:
                    continue
                out_fh.write(_json_dumps_line(pending[paper_id]))
                stats.appended_new += 1

        temp_path.chmod(target_mode)
        temp_path.replace(snapshot_path)
        return stats
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def merge_snapshot_from_db(snapshot_path: Path, updates_db_path: Path) -> MergeStats:
    stats = MergeStats()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    target_mode = _target_file_mode(snapshot_path)

    conn = sqlite3.connect(updates_db_path)
    lookup_cur = conn.cursor()
    matched_ids: set[str] = set()

    fd, temp_name = tempfile.mkstemp(
        prefix=snapshot_path.name + ".",
        suffix=".tmp",
        dir=str(snapshot_path.parent),
        text=True,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with temp_path.open("w", encoding="utf-8") as out_fh:
            if snapshot_path.is_file():
                with snapshot_path.open("r", encoding="utf-8") as in_fh:
                    for line in in_fh:
                        raw_line = line if line.endswith("\n") else line + "\n"
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            obj_any = json.loads(stripped)
                        except Exception:
                            out_fh.write(raw_line)
                            continue
                        if not isinstance(obj_any, dict):
                            out_fh.write(raw_line)
                            continue
                        paper_id = _normalize_text(obj_any.get("id"))
                        if not paper_id:
                            out_fh.write(raw_line)
                            continue

                        row = lookup_cur.execute(
                            "SELECT payload, deleted FROM updates WHERE id = ?",
                            (paper_id,),
                        ).fetchone()
                        if row is None:
                            out_fh.write(raw_line)
                            stats.kept_existing += 1
                            continue

                        matched_ids.add(paper_id)
                        payload, deleted_flag = row
                        if int(deleted_flag):
                            stats.deleted_existing += 1
                            continue

                        update = json.loads(str(payload))
                        merged = dict(obj_any)
                        merged.update(update)
                        out_fh.write(_json_dumps_line(merged))
                        stats.replaced_existing += 1

            append_cur = conn.cursor()
            for paper_id, payload in append_cur.execute(
                "SELECT id, payload FROM updates WHERE deleted = 0 ORDER BY id"
            ):
                if paper_id in matched_ids:
                    continue
                update = json.loads(str(payload))
                out_fh.write(_json_dumps_line(update))
                stats.appended_new += 1

        temp_path.chmod(target_mode)
        temp_path.replace(snapshot_path)
        return stats
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        conn.close()


def _write_state(
    *,
    state_path: Path,
    snapshot_path: Path,
    from_date: Optional[str],
    set_spec: Optional[str],
    result: HarvestResult,
    merge_stats: Optional[MergeStats],
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_url": ARXIV_OAI_BASE_URL,
        "metadata_prefix": "arXivRaw",
        "snapshot_path": str(snapshot_path),
        "set_spec": str(set_spec or ""),
        "requested_from_date": str(from_date or ""),
        "last_response_date": str(result.response_date or ""),
        "max_seen_update_date": str(result.max_update_date or ""),
        "requests_made": int(result.requests_made),
        "records_upserted": int(result.upsert_count),
        "records_deleted": int(result.deleted_count),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    if merge_stats is not None:
        payload["merge_stats"] = {
            "replaced_existing": int(merge_stats.replaced_existing),
            "deleted_existing": int(merge_stats.deleted_existing),
            "appended_new": int(merge_stats.appended_new),
            "kept_existing": int(merge_stats.kept_existing),
        }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh /data/arxiv/arxiv-metadata-oai-snapshot.json from arXiv OAI-PMH."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--base-url", type=str, default=ARXIV_OAI_BASE_URL)
    parser.add_argument("--metadata-prefix", type=str, default="arXivRaw")
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="OAI from-date (YYYY-MM-DD). If omitted, use saved sync state or infer from the local snapshot.",
    )
    parser.add_argument(
        "--set-spec",
        type=str,
        default=None,
        help="Optional OAI set spec such as cs or cs:cs:AI.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Do a full harvest instead of an incremental one. This can take a long time.",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=DEFAULT_OVERLAP_DAYS,
        help="When inferring from an existing snapshot without sync state, rewind this many days for safety.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-records", type=int, default=0, help="Stop after collecting this many updates.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and summarize updates without merging them.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.metadata_prefix != "arXivRaw":
        print("Only --metadata-prefix arXivRaw is currently supported.", file=sys.stderr)
        return 2

    from_date = args.from_date
    if args.full:
        from_date = None
    elif not from_date:
        from_date = infer_from_date(
            snapshot_path=args.output_path,
            state_path=args.state_path,
            overlap_days=args.overlap_days,
        )
        if from_date is None:
            print(
                "Unable to infer a starting date. Use --from-date YYYY-MM-DD or --full for an initial harvest.",
                file=sys.stderr,
            )
            return 2

    print(
        f"Harvesting arXiv metadata from {args.base_url} "
        f"(prefix={args.metadata_prefix}, from={from_date or 'FULL'}, set={args.set_spec or 'ALL'})...",
        file=sys.stderr,
    )

    result = harvest_records(
        base_url=args.base_url,
        metadata_prefix=args.metadata_prefix,
        from_date=from_date,
        set_spec=args.set_spec,
        timeout_seconds=args.timeout_seconds,
        user_agent=DEFAULT_USER_AGENT,
        max_records=max(0, int(args.max_records)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
    )

    print(
        "Fetched "
        f"{result.upsert_count} upserts, {result.deleted_count} deletions "
        f"across {result.requests_made} OAI request(s); "
        f"response_date={result.response_date or '(missing)'}, "
        f"max_update_date={result.max_update_date or '(missing)'}",
        file=sys.stderr,
    )

    try:
        if args.dry_run:
            return 0

        if not result.upsert_count and not result.deleted_count:
            _write_state(
                state_path=args.state_path,
                snapshot_path=args.output_path,
                from_date=from_date,
                set_spec=args.set_spec,
                result=result,
                merge_stats=None,
            )
            print("No snapshot changes detected; state file updated only.", file=sys.stderr)
            return 0

        merge_stats = merge_snapshot_from_db(
            snapshot_path=args.output_path,
            updates_db_path=result.updates_db_path,
        )
        _write_state(
            state_path=args.state_path,
            snapshot_path=args.output_path,
            from_date=from_date,
            set_spec=args.set_spec,
            result=result,
            merge_stats=merge_stats,
        )

        print(
            "Merged snapshot with "
            f"{merge_stats.replaced_existing} replacements, "
            f"{merge_stats.deleted_existing} deletions, "
            f"{merge_stats.appended_new} appended records, "
            f"{merge_stats.kept_existing} unchanged records kept.",
            file=sys.stderr,
        )
        print(f"Wrote sync state to {args.state_path}", file=sys.stderr)
        return 0
    finally:
        try:
            result.updates_db_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
