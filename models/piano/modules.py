"""
PIANO modules: goal generation, skill execution, awareness, talking, social.
These modules are thin wrappers over the MirrorMind tools and local shell actions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from models.mirrormind.context import ContextAssembler
from models.mirrormind.domain import DomainAgent
from models.mirrormind.twins import RepoTwin, PaperTwin
from models.piano.config import resolve_command, load_pipeline_config


class GoalGenerator:
    """Derives subgoals using DomainAgent search."""

    def __init__(self, domain_agent: Optional[DomainAgent] = None) -> None:
        self.domain_agent = domain_agent or DomainAgent()

    def propose(self, task_text: str) -> List[str]:
        concepts = self.domain_agent.search_concepts(task_text, top_k=3)
        names = [c.get("name") for c in concepts if c.get("name")]
        repos = []
        for c in concepts:
            repos.extend(c.get("top_repos", []))
        goals = [f"Investigate concept {n}" for n in names]
        goals += [f"Inspect repo {r}" for r in repos[:2]]
        return goals or [f"Clarify task: {task_text}"]


class SkillExecutor:
    """Skill executor; wraps local shell/file actions with safe defaults."""

    def __init__(
        self,
        base_repo_root: Path = Path("/data/repositories"),
        domain_agent: Optional[DomainAgent] = None,
        ci_runner: Optional[Callable[[Path, Sequence[str]], str]] = None,
        graph_client: Optional[Callable[[str], List[str]]] = None,
        fine_tune_script: Optional[Path] = None,
        inference_script: Optional[Path] = None,
        apply_lora_script: Optional[Path] = None,
        pipeline_config: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.base_repo_root = base_repo_root
        self.domain_agent = domain_agent or DomainAgent()
        self.ci_runner = ci_runner
        self.graph_client = graph_client
        self.fine_tune_script = fine_tune_script
        self.inference_script = inference_script
        self.apply_lora_script = apply_lora_script
        self.pipeline_config = pipeline_config or load_pipeline_config()

    def edit_file(self, repo_id: str, rel_path: str, patch: str) -> str:
        """
        Git-aware edit: append patch and return a diff snippet.
        This is deliberately simple; callers can replace with proper apply_patch.
        """
        repo_path = self.base_repo_root / repo_id
        target = repo_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            before = target.read_text(encoding="utf-8") if target.exists() else ""
            target.write_text(before + "\n" + patch, encoding="utf-8")
            diff = subprocess.run(
                ["git", "diff", "--", rel_path],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            snippet = diff.stdout[-400:] if diff.stdout else "no-diff"
            return f"[edit_file:{repo_id}/{rel_path}] {snippet}"
        except Exception as exc:
            return f"[edit_file_error:{repo_id}/{rel_path}] {exc}"

    def edit_code(self, repo_id: str, hint: str) -> str:
        return self.edit_file(repo_id, "PIANO_EDIT.txt", hint)

    def run_tests(self, repo_id: str) -> str:
        repo_path = self.base_repo_root / repo_id
        cmd = ["pytest", "-q", "--maxfail=1"]
        try:
            proc = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=180)
            status = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"
            return f"[run_tests:{repo_id}] {status} stdout={proc.stdout[-400:]}"
        except Exception as exc:
            return f"[run_tests_error:{repo_id}] {exc}"

    def run_benchmarks(self, repo_id: str, command: Sequence[str]) -> str:
        repo_path = self.base_repo_root / repo_id
        try:
            proc = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, timeout=600)
            status = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"
            return f"[run_benchmarks:{repo_id}] {status} stdout={proc.stdout[-400:]}"
        except Exception as exc:
            return f"[run_benchmarks_error:{repo_id}] {exc}"

    def run_ci(self, repo_id: str, command: Sequence[str] | None = None) -> str:
        """CI adapter stub; default to `make test` if present, else pytest."""
        repo_path = self.base_repo_root / repo_id
        if self.ci_runner:
            try:
                return self.ci_runner(repo_path, command or [])
            except Exception as exc:
                return f"[run_ci_error:{repo_id}] {exc}"
        # Select a sensible default: scripts/ci.sh -> make test -> pytest.
        if command:
            cmd = command
        elif (repo_path / "scripts" / "ci.sh").exists():
            cmd = ["bash", "scripts/ci.sh"]
        elif (repo_path / "Makefile").exists():
            cmd = ["make", "test"]
        else:
            cmd = ["pytest", "-q"]
        try:
            proc = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=600)
            status = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"
            return f"[run_ci:{repo_id}] {status} stdout={proc.stdout[-400:]}"
        except Exception as exc:
            return f"[run_ci_error:{repo_id}] {exc}"

    def read_paper(self, paper_ctx: Dict[str, str]) -> str:
        return f"[read_paper] {paper_ctx.get('episodic_context','')[:200]}"

    def inspect_graph(self, concept: str) -> str:
        return f"[query_graph:{concept}]"

    def graph_neighbors(self, concept: str) -> str:
        """Graph tool adapter using DomainAgent or injected graph client."""
        try:
            if self.graph_client and hasattr(self.graph_client, "expand"):
                neighbors = self.graph_client.expand(concept)  # type: ignore[attr-defined]
                names = [n.get("name") or n.get("concept_id") for n in neighbors if isinstance(n, dict)]
            else:
                neighbors = self.domain_agent.expand_concepts(concept)
                names = [n.get("name") or n.get("concept_id") for n in neighbors]
            return f"[graph_neighbors:{concept}] " + ", ".join([n for n in names if n][:8])
        except Exception as exc:
            return f"[graph_neighbors_error:{concept}] {exc}"

    def update_persona(self, entity_id: str, note: str) -> str:
        return f"[update_persona:{entity_id}] {note}"

    def add_concept_node(self, concept: str, note: str) -> str:
        return f"[add_concept_node:{concept}] {note}"

    def apply_lora(self, adapter_path: str) -> str:
        path = Path(adapter_path)
        if not path.exists():
            return f"[apply_lora_missing:{adapter_path}]"
        try:
            cmd = resolve_command(
                "PIANO_APPLY_LORA_CMD",
                ["python", str(self.apply_lora_script or path)],
                overrides=self.pipeline_config.get("apply_lora_cmd"),
            )
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            status = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"
            return f"[apply_lora:{adapter_path}] {status} stdout={proc.stdout[-200:]}"
        except Exception as exc:
            return f"[apply_lora_error:{adapter_path}] {exc}"

    def fine_tune(self, repo_id: str, config_path: str) -> str:
        cfg = Path(config_path)
        if not cfg.exists():
            return f"[fine_tune_missing:{config_path}]"
        try:
            if self.fine_tune_script and self.fine_tune_script.exists():
                default_cmd = ["python", str(self.fine_tune_script), "--config", str(cfg)]
            else:
                default_cmd = ["python", "-m", "models.cli", "--experiment", str(cfg), "--mode", "train"]
            cmd = resolve_command("PIANO_FINE_TUNE_CMD", default_cmd, overrides=self.pipeline_config.get("fine_tune_cmd"))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            status = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"
            return f"[fine_tune:{repo_id}] {status} stdout={proc.stdout[-400:]}"
        except Exception as exc:
            return f"[fine_tune_error:{repo_id}] {exc}"

    def run_inference(self, repo_id: str, script: str) -> str:
        script_path = Path(script)
        if not script_path.exists() and self.inference_script:
            script_path = self.inference_script
        if not script_path.exists():
            return f"[run_inference_missing:{script}]"
        try:
            cmd = resolve_command("PIANO_INFERENCE_CMD", ["python", str(script_path)], overrides=self.pipeline_config.get("inference_cmd"))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            status = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"
            return f"[run_inference:{repo_id}] {status} stdout={proc.stdout[-400:]}"
        except Exception as exc:
            return f"[run_inference_error:{repo_id}] {exc}"


class ActionAwareness:
    """Tracks recent actions/outcomes."""

    def __init__(self) -> None:
        self.log: List[str] = []

    def record(self, action: str) -> None:
        self.log.append(action)

    def recent(self, k: int = 5) -> List[str]:
        return self.log[-k:]


class TalkingModule:
    """Generates user-facing text via LLM when available."""

    def __init__(self, llm: Optional[Any] = None, logger: Optional[Any] = None) -> None:  # type: ignore[name-defined]
        self.llm = llm
        self._logger = logger

    def respond(self, context: Dict[str, str]) -> str:
        if self.llm:
            semantic = (context.get("semantic_context") or "")[:600]
            episodic = (context.get("episodic_context") or "")[:600]
            prompt = "\n".join(
                [
                    f"Persona:\n{context.get('system','')[:120]}",
                    "Semantic context:\n" + semantic,
                    "Episodic snippets:\n" + episodic,
                    "User task:\n" + context.get("user_task", ""),
                    "Give next steps.\nAnswer:",
                ]
            )
            try:
                raw = (self.llm.generate([prompt], max_new_tokens=96, temperature=0.0) or [""])[0]
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                bullets = []
                for ln in lines:
                    if ln.startswith(("-", "*")):
                        bullets.append(ln)
                    elif ln[:1].isdigit():
                        bullets.append("- " + ln.split(".", 1)[-1].strip())
                    if len(bullets) >= 3:
                        break
                if not bullets and lines:
                    # Fallback to first 3 sentences split by period.
                    sentences = [seg.strip() for seg in raw.split(".") if seg.strip()]
                    bullets = [f"- {s}" for s in sentences[:3]]
                if bullets:
                    return "\n".join(bullets[:3])
                return raw
            except Exception as exc:
                if self._logger:
                    self._logger.warning("LLM generation failed, falling back to template: %s", exc)
        return f"Plan:\n{context.get('semantic_context','')}\nNext:\n{context.get('user_task','')}"


class SocialModule:
    """Placeholder for multi-agent messaging."""

    def broadcast(self, message: str) -> str:
        return f"[broadcast]{message}"


class ContextBuilder:
    """Small helper combining twins and ContextAssembler."""

    def __init__(self, assembler: Optional[ContextAssembler] = None) -> None:
        self.assembler = assembler or ContextAssembler()

    def for_repo(self, twin: RepoTwin, task_text: str) -> Dict[str, str]:
        return self.assembler.build_repo_context(twin, task_text)

    def for_paper(self, twin: PaperTwin, task_text: str) -> Dict[str, str]:
        return self.assembler.build_paper_context(twin, task_text)
