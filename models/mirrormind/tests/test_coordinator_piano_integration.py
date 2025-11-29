from typing import Dict, Any

import pytest

from models.mirrormind.coordinator import Coordinator
from models.mirrormind.domain import DomainGraph, DomainAgent


class DummyPianoAgent:
    def __init__(self, task: str, repo_id=None, paper_id=None) -> None:
        self.task = task
        self.repo_id = repo_id
        self.paper_id = paper_id
        self._calls = 0

    def step(self) -> Dict[str, Any]:
        self._calls += 1
        return {
            "intent": "Plan",
            "rationale": f"step-{self._calls}",
            "action_result": "ok",
        }


@pytest.mark.parametrize("steps", [1, 3])
def test_coordinator_run_multi_step_uses_piano_agent(monkeypatch, steps: int):
    # Patch the PianoAgent used inside the coordinator to avoid touching
    # the real PIANO implementation or filesystem.
    monkeypatch.setattr("models.piano.agent.PianoAgent", DummyPianoAgent)

    graph = DomainGraph()
    agent = DomainAgent(domain_graph=graph, graph_client=None)
    coord = Coordinator(domain_agent=agent)

    out = coord.run_multi_step("refactor subsystem", steps=steps)
    assert out["task"] == "refactor subsystem"
    assert "trajectory" in out and "metrics" in out
    assert len(out["trajectory"]) == steps
    metrics = out["metrics"]
    # Metrics should follow the PAVU-style loop description: actions & rationales tracked.
    assert metrics["steps"] == steps
    assert len(metrics["actions"]) == steps
    assert len(metrics["rationales"]) == steps


