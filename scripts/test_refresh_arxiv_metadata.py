from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.refresh_arxiv_metadata import (
    HarvestBatch,
    infer_from_date,
    merge_snapshot,
    merge_snapshot_from_db,
    parse_oai_list_records,
)


def test_parse_oai_list_records_extracts_updates_and_deletions() -> None:
    xml_text = """\
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-04-20T22:31:27Z</responseDate>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2501.00001</identifier>
        <datestamp>2025-01-03</datestamp>
      </header>
      <metadata>
        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
          <id>2501.00001</id>
          <submitter>Abel Valverde Salamanca</submitter>
          <version version="v1">
            <date>Mon, 07 Oct 2024 10:45:05 GMT</date>
            <size>2113kb</size>
            <source_type>D</source_type>
          </version>
          <title>Mathematical modelling of flow and adsorption in a gas chromatograph</title>
          <authors>A. Cabrera-Codony, A. Valverde</authors>
          <categories>cs.CE physics.chem-ph</categories>
          <comments>35 pages</comments>
          <license>http://creativecommons.org/licenses/by-nc-sa/4.0/</license>
          <abstract>Example abstract.</abstract>
        </arXivRaw>
      </metadata>
    </record>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:2501.99999</identifier>
        <datestamp>2025-01-04</datestamp>
      </header>
    </record>
    <resumptionToken>opaque-token</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""
    batch = parse_oai_list_records(xml_text)

    assert isinstance(batch, HarvestBatch)
    assert batch.response_date == "2026-04-20T22:31:27Z"
    assert batch.resumption_token == "opaque-token"
    assert batch.deleted_ids == {"2501.99999"}
    assert set(batch.records) == {"2501.00001"}

    record = batch.records["2501.00001"]
    assert record["id"] == "2501.00001"
    assert record["submitter"] == "Abel Valverde Salamanca"
    assert record["authors"] == "A. Cabrera-Codony, A. Valverde"
    assert record["categories"] == "cs.CE physics.chem-ph"
    assert record["license"] == "http://creativecommons.org/licenses/by-nc-sa/4.0/"
    assert record["update_date"] == "2025-01-03"
    assert record["versions"] == [
        {
            "version": "v1",
            "created": "Mon, 07 Oct 2024 10:45:05 GMT",
            "size": "2113kb",
            "source_type": "D",
        }
    ]


def test_merge_snapshot_replaces_deletes_and_appends(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    snapshot_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2501.00001",
                        "title": "Old title",
                        "authors": "Old author",
                        "authors_parsed": [["Old", "Author", ""]],
                        "update_date": "2024-12-31",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps({"id": "2501.00002", "title": "Delete me", "update_date": "2024-12-31"}),
                json.dumps({"id": "2501.00003", "title": "Keep me", "update_date": "2024-12-31"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    harvested = {
        "2501.00001": {
            "id": "2501.00001",
            "title": "New title",
            "authors": "New author",
            "update_date": "2025-01-03",
            "versions": [{"version": "v1"}, {"version": "v2"}],
        },
        "2501.00004": {
            "id": "2501.00004",
            "title": "Brand new",
            "authors": "Fresh author",
            "update_date": "2025-01-05",
            "versions": [{"version": "v1"}],
        },
    }
    deleted_ids = {"2501.00002"}

    stats = merge_snapshot(snapshot_path, harvested=harvested, deleted_ids=deleted_ids)

    rows = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["id"] for row in rows] == ["2501.00001", "2501.00003", "2501.00004"]
    assert rows[0]["title"] == "New title"
    assert rows[0]["authors_parsed"] == [["Old", "Author", ""]]
    assert rows[1]["title"] == "Keep me"
    assert rows[2]["title"] == "Brand new"

    assert stats.replaced_existing == 1
    assert stats.deleted_existing == 1
    assert stats.appended_new == 1
    assert stats.kept_existing == 1


def test_merge_snapshot_from_db_replaces_deletes_and_appends(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    db_path = tmp_path / "updates.sqlite3"
    snapshot_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2501.00001",
                        "title": "Old title",
                        "authors_parsed": [["Old", "Author", ""]],
                        "update_date": "2024-12-31",
                    }
                ),
                json.dumps({"id": "2501.00002", "title": "Delete me", "update_date": "2024-12-31"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE updates (
            id TEXT PRIMARY KEY,
            payload TEXT,
            deleted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO updates(id, payload, deleted) VALUES (?, ?, 0)",
        (
            "2501.00001",
            json.dumps({"id": "2501.00001", "title": "New title", "update_date": "2025-01-03"}),
        ),
    )
    conn.execute(
        "INSERT INTO updates(id, payload, deleted) VALUES (?, ?, 1)",
        ("2501.00002", None),
    )
    conn.execute(
        "INSERT INTO updates(id, payload, deleted) VALUES (?, ?, 0)",
        (
            "2501.00003",
            json.dumps({"id": "2501.00003", "title": "Brand new", "update_date": "2025-01-05"}),
        ),
    )
    conn.commit()
    conn.close()

    stats = merge_snapshot_from_db(snapshot_path, updates_db_path=db_path)
    rows = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert [row["id"] for row in rows] == ["2501.00001", "2501.00003"]
    assert rows[0]["title"] == "New title"
    assert rows[0]["authors_parsed"] == [["Old", "Author", ""]]
    assert rows[1]["title"] == "Brand new"
    assert stats.replaced_existing == 1
    assert stats.deleted_existing == 1
    assert stats.appended_new == 1


def test_infer_from_date_prefers_state_then_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    state_path = tmp_path / "arxiv-metadata-oai-state.json"

    snapshot_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "update_date": "2025-11-14", "versions": [{"version": "v1"}]}),
                json.dumps(
                    {
                        "id": "2",
                        "update_date": "",
                        "versions": [{"version": "v1", "created": "Thu, 13 Nov 2025 18:59:57 GMT"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert infer_from_date(snapshot_path, state_path, overlap_days=7) == "2025-11-07"

    state_path.write_text(json.dumps({"last_response_date": "2026-04-20"}) + "\n", encoding="utf-8")
    assert infer_from_date(snapshot_path, state_path, overlap_days=7) == "2026-04-20"
