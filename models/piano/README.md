### PIANO control scaffold for the Code+Paper world

This module provides a minimal PIANO-style agent loop that rides on top of the MirrorMind memory layer:

- `state.py` – shared `AgentState` holding working memory + pointers to RepoTwin/PaperTwin/DomainAgent.
- `controller.py` – `CognitiveController` coherence bottleneck that emits a high-level `Intent`.
- `modules.py` – stub modules for goal generation, skill execution, action awareness, talking, social, and context building via MirrorMind.
- `agent.py` – `PianoAgent` that ties everything together: summarize state → decide intent → call module → log awareness.

Usage (stub):
```python
from models.piano import PianoAgent

agent = PianoAgent(task="fix failing optimizer tests", repo_id="AgentLab")
step_out = agent.step()
print(step_out)
```

Key pieces now implemented:
- `SkillExecutor` wraps git-aware file edits, pytest runs, benchmark commands, persona/concept updates, and hooks for apply_lora/fine_tune/run_inference; swap with your preferred tooling for production.
- SkillExecutor accepts optional `ci_runner` and `graph_client` callbacks so you can plug in real CI pipelines or graph DB queries; defaults fall back to scripts/ci.sh → make test → pytest and DomainAgent/ConceptGraph expansion. You can also pass `fine_tune_script`, `inference_script`, and `apply_lora_script` to route those actions to your HF/PEFT runners (or set env vars `PIANO_FINE_TUNE_CMD`, `PIANO_INFERENCE_CMD`, `PIANO_APPLY_LORA_CMD`, or provide `pipeline_config={"fine_tune_cmd":[...], ...}` / `PIANO_PIPELINE_CONFIG=/path/cmds.json`), otherwise it uses the built-in CLI or direct script paths.
- `CognitiveController` accepts an optional policy callable; by default it uses the Llama 1B-backed `LLMIntentPolicy` to pick intents (heuristic fallback available) and supports intents like `spawn_specialist`, `run_benchmarks`, `update_persona`, `add_concept_node`, `apply_lora`, `fine_tune`, `run_inference`.
- `GoalGenerator` uses DomainAgent search to propose concept/repo subgoals.
- `TalkingModule` uses the default Llama 1B (cached in `/data/checkpoints`) via `safe_build_llm` when available.
- `smoke.py` exercises a single-agent step.
- `coordinator.run_multi_step` runs a short PianoAgent trajectory and returns metrics; tests under `models/piano/tests/`.

Swap the controller for an LLM policy and connect SkillExecutor to real tools (git-aware edits, CI runners, graph DB) to move beyond the stubs. Multi-agent “civilization” experiments can wrap multiple `PianoAgent` instances and coordinate them via the SocialModule.
