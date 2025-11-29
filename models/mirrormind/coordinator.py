"""
Coordinator and ReviewAgent scaffolding for the MirrorMind orchestration layer.
Routes tasks through DomainAgent, RepoTwins, and PaperTwins, and assembles contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any

from models.mirrormind.context import ContextAssembler
from models.mirrormind.domain import DomainAgent, DomainGraph
from models.mirrormind.graph_client import FileGraphClient
from models.mirrormind.graph_neo4j import Neo4jGraphClient  # optional
from models.mirrormind.twins import RepoTwin, PaperTwin


@dataclass
class TaskDescriptor:
    task_id: str
    task_text: str
    task_type: str = "code_change"
    constraints: Dict[str, str] = field(default_factory=dict)


class ReviewAgent:
    """Simple review synthesizer that merges partial plans."""

    def _extract_dependencies(self, plan: Dict[str, Any]) -> List[str]:
        """Best-effort extraction of dependencies from a plan."""
        deps = plan.get("dependencies") or []
        if isinstance(deps, list):
            return [str(d) for d in deps]
        if isinstance(deps, dict):
            return [f"{k}:{v}" for k, v in deps.items()]
        return []

    def _detect_conflicts(self, actions: Sequence[str]) -> List[str]:
        """
        Heuristic conflict detector:
        - flags actions that mention rollback and apply in the same span
        - flags actions that target the same repo/file with contradictory verbs
        """
        conflicts: List[str] = []
        lower = [a.lower() for a in actions]
        for act in lower:
            if "rollback" in act and "apply" in act:
                conflicts.append(f"possible_rollback_conflict:{act}")
        return conflicts

    def _uncertainty_flags_for_plan(self, plan: Dict[str, Any]) -> List[str]:
        """
        Emit lightweight uncertainty flags based on textual cues in actions
        and evidence.
        """
        flags: List[str] = []
        text_bits: List[str] = []
        for key in ("actions", "evidence"):
            for v in plan.get(key, []):
                text_bits.append(str(v))
        blob = " ".join(text_bits).lower()
        if any(tok in blob for tok in ("todo", "tbd", "unknown", "uncertain", "maybe")):
            flags.append("plan_contains_todo_or_uncertain_cues")
        if "missing" in blob:
            flags.append("plan_refers_to_missing_artifacts")
        return flags

    def synthesize(self, repo_plans: Sequence[Dict[str, Any]], paper_plans: Sequence[Dict[str, Any]]) -> Dict[str, object]:
        """
        Merge repo/paper plans into a global view with:
        - de-duplicated, scored actions
        - evidence map keyed by action index
        - basic consistency report and uncertainty flags

        This is intentionally lightweight but matches the variables described
        in models/paper.md Section 5.2.
        """
        all_plans: List[Dict[str, Any]] = list(repo_plans) + list(paper_plans)
        actions: List[str] = []
        evidence: List[Any] = []
        dependencies: Dict[str, List[str]] = {}
        uncertainty_flags: List[str] = []

        for idx, plan in enumerate(all_plans):
            plan_actions = [str(a) for a in plan.get("actions", [])]
            actions.extend(plan_actions)
            evidence.extend(plan.get("evidence", []))
            deps = self._extract_dependencies(plan)
            if deps:
                dependencies[f"plan_{idx}"] = deps
            uncertainty_flags.extend(self._uncertainty_flags_for_plan(plan))

        evidence_map: Dict[str, Any] = {}
        for idx, ev in enumerate(evidence):
            evidence_map[f"step_{idx}"] = ev

        issues: List[str] = []
        scored_actions: List[Dict[str, Any]] = []
        seen_actions = set()
        for act in actions:
            if act in seen_actions:
                continue
            seen_actions.add(act)
            score = 1.0
            act_lower = act.lower()
            if "missing" in act_lower or "error" in act_lower:
                issues.append(f"action_issue:{act}")
                score = 0.2
            if "todo" in act_lower or "tbd" in act_lower:
                issues.append(f"action_incomplete:{act}")
                score = 0.3
            scored_actions.append({"action": act, "score": score})
        scored_actions = sorted(scored_actions, key=lambda x: x["score"], reverse=True)

        conflicts = self._detect_conflicts([a["action"] for a in scored_actions])
        issues.extend(conflicts)

        consistency_report = "no-actions"
        if scored_actions:
            if conflicts:
                consistency_report = "conflicts-detected"
            elif issues:
                consistency_report = "issues-detected"
            else:
                consistency_report = "consistent"

        evidence_summary = [ev for ev in evidence[:5]]
        # De-duplicate uncertainty flags while preserving order.
        seen_flags = set()
        uniq_flags: List[str] = []
        for f in uncertainty_flags:
            if f in seen_flags:
                continue
            seen_flags.add(f)
            uniq_flags.append(f)

        return {
            "global_plan": [a["action"] for a in scored_actions],
            "evidence_map": evidence_map,
            "evidence_summary": evidence_summary,
            "dependencies": dependencies,
            "consistency_report": consistency_report,
            "issues": issues,
            "uncertainty_flags": uniq_flags,
        }


class Coordinator:
    """Interdisciplinary orchestrator."""

    def __init__(
        self,
        domain_agent: Optional[DomainAgent] = None,
        context_assembler: Optional[ContextAssembler] = None,
        review_agent: Optional[ReviewAgent] = None,
    ) -> None:
        if domain_agent:
            self.domain_agent = domain_agent
        else:
            graph = DomainGraph()
            client = FileGraphClient(graph)
            self.domain_agent = DomainAgent(domain_graph=graph, graph_client=client)
        self.context_assembler = context_assembler or ContextAssembler()
        self.review_agent = review_agent or ReviewAgent()
        self.last_concepts: List[Dict[str, object]] = []
        self.last_domains: List[str] = []

    def _score_entities_from_concepts(
        self, concepts: Sequence[Dict[str, object]], entity_type: str, limit: int = 3
    ) -> List[str]:
        """
        Use concept search results to pull experts from the DomainAgent and
        aggregate scores. Falls back to top_repos / top_papers if expert API
        returns nothing.
        """
        scores: Dict[str, float] = {}
        fallback: List[Tuple[str, float]] = []
        for c in concepts:
            cid = c.get("concept_id")
            base = float(c.get("score") or 0.0)
            if not cid:
                continue
            experts = self.domain_agent.get_expert_entities_for_concept(str(cid), k=limit * 2)
            for ent_id, ent_type, ent_score in experts:
                if ent_type != entity_type or not ent_id:
                    continue
                score = 0.6 * base + 0.4 * float(ent_score)
                scores[ent_id] = max(scores.get(ent_id, 0.0), score)
            # collect fallback candidates to remain backwards compatible
            tops = c.get("top_repos" if entity_type == "repo" else "top_papers", []) or []
            for ent_id in tops:
                if ent_id:
                    fallback.append((ent_id, base))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked and fallback:
            ranked = sorted(fallback, key=lambda x: x[1], reverse=True)

        selected: List[str] = []
        for ent_id, _ in ranked:
            if ent_id not in selected:
                selected.append(ent_id)
            if len(selected) >= limit:
                break
        return selected

    def _build_repo_plans(self, repos: Sequence[str], task_text: str) -> List[Dict[str, object]]:
        plans: List[Dict[str, object]] = []
        for repo_id in repos:
            twin = RepoTwin(repo_id=repo_id)
            ctx = self.context_assembler.build_repo_context(twin, task_text)
            plans.append({"repo_id": repo_id, "actions": [f"Consult repo {repo_id}"], "evidence": [ctx]})
        return plans

    def _build_paper_plans(self, papers: Sequence[str], task_text: str) -> List[Dict[str, object]]:
        plans: List[Dict[str, object]] = []
        for paper_id in papers:
            twin = PaperTwin(paper_id=paper_id)
            ctx = self.context_assembler.build_paper_context(twin, task_text)
            plans.append({"paper_id": paper_id, "actions": [f"Read paper {paper_id}"], "evidence": [ctx]})
        return plans

    def run(self, task: TaskDescriptor) -> Dict[str, object]:
        concepts = self.domain_agent.search_concepts(task.task_text, top_k=6)
        self.last_concepts = concepts
        self.last_domains = [c.get("name", "") for c in concepts if c.get("name")]
        selected_repos = self._score_entities_from_concepts(concepts, entity_type="repo", limit=3)
        selected_papers = self._score_entities_from_concepts(concepts, entity_type="paper", limit=3)
        repo_plans = self._build_repo_plans(selected_repos, task.task_text)
        paper_plans = self._build_paper_plans(selected_papers, task.task_text)
        integrated = self.review_agent.synthesize(repo_plans, paper_plans)
        return {
            "task": task.task_text,
            "concepts": concepts,
            "selected_repos": selected_repos,
            "selected_papers": selected_papers,
            "repo_plans": repo_plans,
            "paper_plans": paper_plans,
            "integrated": integrated,
        }

    def run_with_piano(self, task_text: str, repo_id: Optional[str] = None, paper_id: Optional[str] = None) -> Dict[str, object]:
        """Convenience entry to run a single PianoAgent step for a task."""
        # Lazy import to avoid circular dependency at module import time.
        from models.piano.agent import PianoAgent  # type: ignore

        agent = PianoAgent(task=task_text, repo_id=repo_id, paper_id=paper_id)
        step_out = agent.step()
        return {"task": task_text, "repo_id": repo_id, "paper_id": paper_id, "step": step_out}

    def run_multi_step(
        self,
        task_text: str,
        steps: int = 3,
        repo_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        max_retries: int = 0,
    ) -> Dict[str, object]:
        """Execute a small multi-step loop with a PianoAgent and collect metrics."""
        from models.piano.agent import PianoAgent  # type: ignore

        # If no repo/paper specified, pick top from search.
        if repo_id is None or paper_id is None:
            concepts = self.domain_agent.search_concepts(task_text, top_k=6)
            if repo_id is None:
                repos = self._score_entities_from_concepts(concepts, entity_type="repo", limit=1)
                if repos:
                    repo_id = repos[0]
            if paper_id is None:
                papers = self._score_entities_from_concepts(concepts, entity_type="paper", limit=1)
                if papers:
                    paper_id = papers[0]

        agent = PianoAgent(task=task_text, repo_id=repo_id, paper_id=paper_id)
        trajectory: List[Dict[str, object]] = []
        attempts = max(1, steps)
        retries = max(0, max_retries)
        while attempts > 0:
            out = agent.step()
            trajectory.append(out)
            attempts -= 1
            if retries > 0 and "missing" in str(out.get("action_result", "")).lower():
                retries -= 1
                attempts += 1  # retry once for missing actions
        metrics = {
            "actions": [t["intent"] for t in trajectory],
            "unique_actions": len(set(t["intent"] for t in trajectory)),
            "rationales": [t.get("rationale") for t in trajectory],
            "steps": len(trajectory),
        }
        return {"task": task_text, "trajectory": trajectory, "metrics": metrics}
