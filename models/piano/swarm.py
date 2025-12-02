"""
Multi-agent PIANO swarm orchestration over the MirrorMind graph.

This module provides a very lightweight "civilization" layer that:

- spins up multiple PianoAgent instances with optional roles and repo/paper
  specializations,
- runs short trajectories for each agent,
- tracks intent distributions, simple specialization scores, rule violations,
  and coarse "meme" frequencies in actions/results.

The goal is not to simulate a rich society but to expose the instrumentation
needed to study parallel information aggregation and division of labor across
repos/papers, in the spirit of the PIANO civilization benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import math

from models.piano.agent import PianoAgent
from models.piano.controller import Intent


@dataclass
class AgentRun:
    agent_id: str
    role: str
    repo_id: Optional[str]
    paper_id: Optional[str]
    intents: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


@dataclass
class SwarmMetrics:
    """Summary statistics for a swarm rollout."""

    role_intent_counts: Dict[str, Dict[str, int]]
    agent_specialization: Dict[str, float]
    rule_violations: List[str]
    meme_counts: Dict[str, int]


class PianoSwarm:
    """
    Orchestrate a small swarm of PianoAgents over the Code+Paper graph.

    Each agent can be given:
      - a role label (e.g., "optimizer_engineer", "paper_summarizer"),
      - an optional (repo_id, paper_id) focus.

    Simple rule sets allow you to specify, per role, which intents are allowed;
    violations are recorded in metrics.
    """

    def __init__(
        self,
        task: str,
        agents: Sequence[Dict[str, Optional[str]]],
        *,
        role_rules: Optional[Dict[str, Sequence[Intent]]] = None,
    ) -> None:
        """
        Args:
            task: Global task text shared by the swarm.
            agents: Sequence of specs with keys:
                - "role": human-readable role label.
                - "repo_id": optional repo focus.
                - "paper_id": optional paper focus.
            role_rules: Optional mapping role -> allowed intents.
        """
        self.task = task
        self.role_rules = role_rules or {}
        self._runs: List[AgentRun] = []
        self._agents: List[PianoAgent] = []
        self._roles: List[str] = []
        self._ids: List[str] = []

        for idx, spec in enumerate(agents):
            role = str(spec.get("role") or f"agent_{idx}")
            repo_id = spec.get("repo_id")
            paper_id = spec.get("paper_id")
            agent = PianoAgent(task=task, repo_id=repo_id, paper_id=paper_id)
            self._agents.append(agent)
            self._roles.append(role)
            self._ids.append(f"{role}-{idx}")
            self._runs.append(
                AgentRun(
                    agent_id=self._ids[-1],
                    role=role,
                    repo_id=repo_id,
                    paper_id=paper_id,
                )
            )

    def run_rounds(self, steps_per_agent: int = 3) -> SwarmMetrics:
        """Execute a small trajectory for each agent and return aggregate metrics."""
        rule_violations: List[str] = []
        meme_counts: Dict[str, int] = {}

        for agent, role, run in zip(self._agents, self._roles, self._runs):
            allowed = set(self.role_rules.get(role, []))
            for _ in range(max(1, steps_per_agent)):
                out = agent.step()
                intent = str(out.get("intent") or "unknown")
                result = str(out.get("action_result") or "")
                run.intents.append(intent)
                run.actions.append(result)
                # Rule tracking.
                if allowed and intent not in allowed:
                    rule_violations.append(f"{run.agent_id}:{intent}")
                # Very crude "meme" tracking: count frequent tokens in actions.
                for tok in result.lower().split():
                    if len(tok) < 4:
                        continue
                    meme_counts[tok] = meme_counts.get(tok, 0) + 1

        role_intent_counts: Dict[str, Dict[str, int]] = {}
        for run in self._runs:
            counts = role_intent_counts.setdefault(run.role, {})
            for intent in run.intents:
                counts[intent] = counts.get(intent, 0) + 1

        # Specialization: for each agent, compute 1 - (entropy / log K), where
        # K is the number of distinct intents this swarm used. Values close to 1
        # indicate that the agent strongly prefers a small subset of intents.
        all_intents: List[str] = []
        for run in self._runs:
            all_intents.extend(run.intents)
        distinct_intents = set(all_intents) or {"talk"}
        k = float(len(distinct_intents))

        def _entropy(counts: Dict[str, int]) -> float:
            total = float(sum(counts.values()) or 1.0)
            h = 0.0
            for c in counts.values():
                p = c / total
                if p > 0.0:
                    h -= p * math.log(p + 1e-12)
            return h

        agent_specialization: Dict[str, float] = {}
        for run in self._runs:
            local_counts: Dict[str, int] = {}
            for intent in run.intents:
                local_counts[intent] = local_counts.get(intent, 0) + 1
            h = _entropy(local_counts)
            norm = math.log(k + 1e-12)
            spec = 0.0 if norm <= 0.0 else 1.0 - min(1.0, h / norm)
            agent_specialization[run.agent_id] = spec

        return SwarmMetrics(
            role_intent_counts=role_intent_counts,
            agent_specialization=agent_specialization,
            rule_violations=rule_violations,
            meme_counts=meme_counts,
        )


