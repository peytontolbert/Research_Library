from pathlib import Path
import json

from models.mirrormind.domain import DomainGraph, DomainAgent
from models.mirrormind.graph_client import FileGraphClient


def test_domain_graph_load_repo_and_paper_concepts(tmp_path: Path):
    repo_concepts = tmp_path / "repo_concepts.jsonl"
    paper_concepts = tmp_path / "paper_concepts.jsonl"
    with repo_concepts.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "name": "foo concept", "repo_id": "r1"}) + "\n")
        f.write(json.dumps({"id": "c2", "name": "bar concept", "repo_id": "r1"}) + "\n")
    with paper_concepts.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c3", "name": "baz concept", "paper_id": "p1"}) + "\n")
        f.write(json.dumps({"id": "c1", "name": "foo concept", "paper_id": "p1"}) + "\n")

    dg = DomainGraph(repo_concepts_path=repo_concepts, paper_concepts_path=paper_concepts)

    # Search should use both token overlap and embeddings and return concept metadata.
    res = dg.search("foo", top_k=2)
    assert res
    top_node, score = res[0]
    assert isinstance(score, float)
    assert top_node.id == "c1"
    assert top_node.embedding  # embedding vector should be populated

    # Repo-based neighbors / edge types.
    neighbors = dg.expand("c1")
    assert neighbors
    neighbor_ids = {n.id for n in neighbors}
    assert "c2" in neighbor_ids  # co-occurs in same repo
    assert "appears_in_same_repo_as" in dg.nodes["c1"].edge_types.get("c2", [])

    # Paper-based neighbors / edge types.
    assert "c3" in dg.nodes["c1"].neighbors
    assert "appears_in_same_paper_as" in dg.nodes["c1"].edge_types.get("c3", [])


def test_file_graph_client_and_domain_agent_shape(tmp_path: Path):
    repo_concepts = tmp_path / "repo_concepts.jsonl"
    with repo_concepts.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "name": "foo", "repo_id": "r1"}) + "\n")
    client = FileGraphClient(DomainGraph(repo_concepts_path=repo_concepts))
    agent = DomainAgent(graph_client=client)
    results = agent.search_concepts("foo")
    assert results
    first = results[0]
    # Ensure we expose the tool variables described in the paper.
    assert "concept_id" in first
    assert "name" in first
    assert "score" in first
    assert "neighbors" in first
    assert "top_repos" in first
    assert "top_papers" in first
