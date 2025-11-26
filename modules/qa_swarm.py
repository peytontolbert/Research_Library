from __future__ import annotations

"""
High-level QA swarm orchestration.

This module provides light-weight, modular interfaces for:

- SemanticRouter: decide which repo/skill/mode to use for a question.
- SkillAdapterManager: plan via RepoLibrary and fetch adapter metadata.
- RetrieverAgent: delegate retrieval + answering to a pluggable function.
- QASwarmController: orchestrate the full QA flow end-to-end.

The intent is to mirror the conceptual "planner / router / retriever /
answerer" architecture while keeping the implementation thin and
grounded in the existing RepoLibrary + QA runtime.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from scripts.repo_library import QueryMode  # type: ignore


@dataclass
class SemanticRoute:
    """
    Routed decision for a QA interaction.
    """

    repo_id: str
    skill: str
    qa_mode: Optional[str]


@dataclass
class SemanticRouter:
    """
    Simple semantic router for QA.

    For now this is a thin wrapper that:
    - Assumes the frontend/UI has already selected the target repo.
    - Always routes to the "qa" skill.

    This is intentionally small so that more advanced routing logic
    (multi-repo, multi-skill, task classification) can be plugged in
    later without changing callers.
    """

    def route(
        self,
        question: str,
        *,
        repo_hint: str,
        qa_mode: Optional[str],
    ) -> SemanticRoute:
        _ = question  # reserved for future routing based on content
        return SemanticRoute(repo_id=repo_hint, skill="qa", qa_mode=qa_mode)


@dataclass
class SkillAdapterManager:
    """
    Bridge between RepoLibrary (planning) and adapter metadata.

    Given a routed QA request, this:
    - Calls `RepoLibrary.query` in `QueryMode.QA`.
    - Validates that the plan is single-repo and matches the routed repo.
    - Extracts the QA adapter metadata for that repo.
    """

    repo_library: Any

    def plan_and_get_qa_meta(
        self,
        *,
        question: str,
        route: SemanticRoute,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        plan = self.repo_library.query(
            question=question,
            mode=QueryMode.QA,
            repo_hint=route.repo_id,
            qa_mode=route.qa_mode,
        )

        repos = plan.get("repos") or []
        if not isinstance(repos, list) or not repos:
            raise ValueError(
                "QA execution requires at least one target repo in the plan."
            )
        if len(repos) != 1 or repos[0] != route.repo_id:
            raise ValueError(
                "QA execution expects a single-repo QA plan matching the routed repo."
            )

        skills_any = plan.get("skills") or {}
        skills: Dict[str, Any] = skills_any if isinstance(skills_any, dict) else {}
        repo_skills_any = skills.get(route.repo_id) or {}
        repo_skills: Dict[str, Any] = (
            repo_skills_any if isinstance(repo_skills_any, dict) else {}
        )
        qa_meta_any = repo_skills.get("qa") or {}
        qa_meta: Dict[str, Any] = qa_meta_any if isinstance(qa_meta_any, dict) else {}
        if not qa_meta:
            raise ValueError(
                f"QA skill for repo_id={route.repo_id!r} is not built or "
                "missing from the adapter registry. Build the QA skill first."
            )
        return plan, qa_meta


@dataclass
class RetrieverAgent:
    """
    Retriever/answering facade used by the swarm controller.

    This delegates to a pluggable `retrieve_fn(plan, qa_meta)` function
    so that existing QA logic (retrieval + prompting + model call) can
    be reused without duplicating code.
    """

    retrieve_fn: Callable[[Dict[str, Any], Dict[str, Any]], str]

    def run(self, plan: Dict[str, Any], qa_meta: Dict[str, Any]) -> str:
        return self.retrieve_fn(plan, qa_meta)


@dataclass
class QASwarmController:
    """
    High-level orchestrator for single-repo QA interactions.

    It wires together:
    - SemanticRouter
    - SkillAdapterManager
    - RetrieverAgent

    and produces the final `skill_chat_result` payload expected by
    `/api/skill_chat` and `/api/qa_execute`.
    """

    router: SemanticRouter
    adapter_manager: SkillAdapterManager
    retriever: RetrieverAgent

    def run_qa(
        self,
        *,
        question: str,
        repo_hint: str,
        qa_mode: Optional[str],
    ) -> Dict[str, Any]:
        route = self.router.route(question, repo_hint=repo_hint, qa_mode=qa_mode)
        plan, qa_meta = self.adapter_manager.plan_and_get_qa_meta(
            question=question,
            route=route,
        )
        answer_text = self.retriever.run(plan, qa_meta)
        return {
            "type": "skill_chat_result",
            "status": "completed",
            "skill": route.skill,
            "plan": plan,
            "answer": answer_text,
        }



