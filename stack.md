### Stack Overview

**Purpose**  
Unified memory + control stack for the Code+Paper world: MirrorMind (memory/substrate) + PIANO (control/agents) + tooling hooks (CI/graph/HF).

**Layers**
- **Data/Graphs**: ProgramGraph, RepoGraph, PaperGraph, ConceptGraph (exports under `exports/`, repo concepts in `models/exports/`).
- **MirrorMind Memory** (`models/mirrormind/`): episodic/semantic stores (`memory.py` with optional dense/sparse/FAISS + `embeddings.py`), persona schemas (`persona.py`), twins (`twins.py`), domain graph/tools (`domain.py` with file/Neo4j clients), context assembly (`context.py`), orchestrator stubs (`coordinator.py`), LLM loader (`llm.py`).
- **PIANO Control** (`models/piano/`): AgentState, CognitiveController (LLM or heuristic intent), LLMIntentPolicy, modules (GoalGenerator, SkillExecutor, TalkingModule, ActionAwareness, SocialModule), PianoAgent, multi-step coordinator entrypoints, smoke/tests.

**Key Integrations**
- **LLM**: default `meta-llama/Llama-3.2-1B-Instruct` cached in `/data/checkpoints`, via shared HF wrappers; Talking/Intent use it when available with deterministic prompts.
- **SkillExecutor Hooks**: git-aware edits, pytest/benchmarks, CI runner hook (scripts/ci.sh → make test → pytest fallback or injected `ci_runner`), graph client hook (DomainAgent fallback or injected `graph_client`), apply_lora/fine_tune/run_inference configurable via scripts/env vars/pipeline_config.
- **Graph Tools**: DomainAgent over repo concepts; `graph_neighbors` adapter can be replaced with real graph DB client.
- **Coordinator**: single-step and multi-step wrappers around PianoAgent; metrics collected per trajectory.

**Status**
- Working smoke: `python -m models.piano.smoke` returns concise 3-step suggestions using Llama 1B instruct.
- Tests: SkillExecutor edits/CI/graph, TalkingModule bullet truncation, basic agent step.
- Gaps vs MirrorMind spec: vector/sparse indexes exist but are heuristic (FAISS/BM25 + text embedder); DomainGraph limited to exported concepts (run `extract_paper_concepts.py` to add papers and load into Neo4j/FileGraph); Coordinator/review layer is stubby; persona/style extraction is heuristic; apply_lora/fine_tune/run_inference need real pipelines.

**Next Steps (suggested)**
1) Swap memory stores to real vector/sparse backends with persistence.  
2) Ingest full ConceptGraph (papers + repos) and hook DomainAgent/graph adapters to a real graph DB (FileGraphClient + Neo4j client provided; run `extract_paper_concepts.py` to seed papers).  
3) Expand Coordinator to real expert selection/review with evidence maps and retries (auto-selects repos/papers via concept search; basic retry and action scoring).  
4) Wire SkillExecutor stubs to actual CI/graph/HF pipelines; add more trajectory tests and role/rule/meme metrics.
