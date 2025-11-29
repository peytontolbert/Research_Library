"""
PIANO agent wrapper tying together state, controller, and modules.
Designed to operate over the Code+Paper environment using MirrorMind twins.
"""

from __future__ import annotations

from typing import Dict, Optional
import uuid

from models.piano.state import AgentState
from models.piano.controller import CognitiveController, Intent
from models.piano.modules import (
    GoalGenerator,
    SkillExecutor,
    ActionAwareness,
    TalkingModule,
    SocialModule,
    ContextBuilder,
)
from models.mirrormind.twins import RepoTwin, PaperTwin
from models.mirrormind.llm import safe_build_llm
from models.piano.policy import LLMIntentPolicy
from models.mirrormind.graph_client import FileGraphClient
from models.mirrormind.graph_neo4j import Neo4jGraphClient  # optional
from models.mirrormind.domain import DomainGraph
import logging


class PianoAgent:
    """Minimal PIANO agent loop."""

    def __init__(
        self,
        task: str,
        repo_id: Optional[str] = None,
        paper_id: Optional[str] = None,
    ) -> None:
        repo_twin = RepoTwin(repo_id) if repo_id else None
        paper_twin = PaperTwin(paper_id) if paper_id else None
        self.state = AgentState(task=task, repo_twin=repo_twin, paper_twin=paper_twin)
        self.logger = logging.getLogger(__name__)
        llm, _, _ = safe_build_llm(model_id="U1")
        if llm is None:
            self.logger.warning("PianoAgent running without LLM; TalkingModule will use template responses.")
        policy = LLMIntentPolicy(llm, allowed=list(Intent.__args__)) if llm else None  # type: ignore[attr-defined]
        self.controller = CognitiveController(policy=policy)
        # Wire a file-backed graph client by default.
        graph = DomainGraph()
        graph_client = FileGraphClient(graph)
        # If Neo4j is available, prefer it.
        try:
            graph_client = Neo4jGraphClient()  # type: ignore
        except Exception:
            pass
        self.state.domain_agent = DomainAgent(domain_graph=graph, graph_client=graph_client)
        self.goals = GoalGenerator(self.state.domain_agent)
        self.skills = SkillExecutor(domain_agent=self.state.domain_agent, graph_client=getattr(graph_client, "expand", None) and graph_client)
        self.awareness = ActionAwareness()
        self.talk = TalkingModule(llm=llm, logger=self.logger)
        self.social = SocialModule()
        self.ctx_builder = ContextBuilder()

    def step(self) -> Dict[str, str]:
        summary = self.state.summarize()
        intent: Intent = self.controller.decide(summary)
        action_result = ""
        rationale = self.controller.last_rationale

        if intent == "plan_more":
            proposed = self.goals.propose(self.state.task)
            self.state.working_memory.extend(proposed)
            action_result = f"proposed_goals:{proposed}"
        elif intent == "inspect_graph":
            action_result = self.skills.inspect_graph(self.state.task)
        elif intent == "edit_code" and self.state.repo_twin:
            ctx = self.ctx_builder.for_repo(self.state.repo_twin, self.state.task)
            action_result = self.skills.edit_code(self.state.repo_twin.entity_id, ctx.get("user_task", ""))
        elif intent == "run_tests" and self.state.repo_twin:
            action_result = self.skills.run_tests(self.state.repo_twin.entity_id)
        elif intent == "read_paper" and self.state.paper_twin:
            ctx = self.ctx_builder.for_paper(self.state.paper_twin, self.state.task)
            action_result = self.skills.read_paper(ctx)
        elif intent == "run_benchmarks" and self.state.repo_twin:
            action_result = self.skills.run_benchmarks(self.state.repo_twin.entity_id, ["pytest", "-q"])
        elif intent == "update_persona" and self.state.repo_twin:
            action_result = self.skills.update_persona(self.state.repo_twin.entity_id, "refresh persona from recent edits")
        elif intent == "add_concept_node":
            action_result = self.skills.add_concept_node(self.state.task, "add concept placeholder")
        elif intent == "apply_lora":
            action_result = self.skills.apply_lora("/data/checkpoints/lora-adapter")
        elif intent == "fine_tune" and self.state.repo_twin:
            action_result = self.skills.fine_tune(self.state.repo_twin.entity_id, "models/experiments/u1_unified_knowledge_model.json")
        elif intent == "run_inference" and self.state.repo_twin:
            action_result = self.skills.run_inference(self.state.repo_twin.entity_id, "python inference.py")
        elif intent == "spawn_specialist":
            action_result = self.social.broadcast(f"spawn specialist for {self.state.task}")
        else:
            # default to talking back
            ctx = {}
            if self.state.repo_twin:
                ctx = self.ctx_builder.for_repo(self.state.repo_twin, self.state.task)
            action_result = self.talk.respond(ctx)

        self.awareness.record(f"{intent}:{action_result}")
        self.state.working_memory.append(f"{intent}:{action_result}")
        return {
            "intent": intent,
            "action_result": action_result,
            "state_id": str(uuid.uuid4()),
            "recent_actions": self.awareness.recent(),
            "rationale": rationale,
        }
