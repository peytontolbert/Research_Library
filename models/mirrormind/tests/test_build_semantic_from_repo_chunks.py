from pathlib import Path
import json

from models.mirrormind.scripts.build_semantic_from_repo_chunks import (
    _repo_id_from_path,
    build_semantic_from_repo_chunks,
)


def test_repo_id_from_path():
    p = "/data/repositories/foo/bar/file.py"
    assert _repo_id_from_path(p) == "foo"
    p2 = "/tmp/foo/bar.py"
    assert _repo_id_from_path(p2) == "foo"


def test_build_semantic_from_repo_chunks(tmp_path: Path):
    chunks_dir = tmp_path / "repos_chunks"
    chunks_dir.mkdir()
    chunk_file = chunks_dir / "repo_chunks_00001.jsonl"
    with chunk_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"path": "/data/repositories/foo/a.py", "code": "alpha beta"}) + "\n")
        f.write(json.dumps({"path": "/data/repositories/bar/b.py", "code": "gamma delta"}) + "\n")

    out = tmp_path / "semantic.jsonl"
    semantic_store = build_semantic_from_repo_chunks(
        chunks_dir,
        out,
        summarize_fn=lambda text: f"summary:{text.split()[0]}",
    )
    # Should produce one summary per repo
    assert len(semantic_store._by_entity.get("foo", [])) == 1
    assert len(semantic_store._by_entity.get("bar", [])) == 1
    assert out.exists()
