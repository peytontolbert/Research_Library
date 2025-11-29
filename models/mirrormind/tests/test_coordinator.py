from pathlib import Path
import json

from models.mirrormind.coordinator import Coordinator, TaskDescriptor
from models.mirrormind.domain import DomainGraph, DomainAgent


def test_coordinator_prefers_expert_entities(tmp_path: Path):
    repo_concepts = tmp_path / "repo_concepts.jsonl"
    # Duplicate concept occurrence to give r1 higher frequency.
    with repo_concepts.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "name": "foo", "repo_id": "r1"}) + "\n")
        f.write(json.dumps({"id": "c1", "name": "foo", "repo_id": "r1"}) + "\n")
        f.write(json.dumps({"id": "c2", "name": "bar", "repo_id": "r2"}) + "\n")

    graph = DomainGraph(
        repo_concepts_path=repo_concepts,
        paper_concepts_path=tmp_path / "paper_concepts.jsonl",
        paper_repo_align_path=tmp_path / "align.jsonl",
    )
    agent = DomainAgent(domain_graph=graph, graph_client=None)
    coord = Coordinator(domain_agent=agent)

    result = coord.run(TaskDescriptor(task_id="t1", task_text="foo thing"))
    assert result["selected_repos"][:1] == ["r1"]
    assert coord.last_concepts
