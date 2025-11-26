## QA Runtime and Model Pipeline

### HTTP Entry Points

- `/api/skill_chat` (POST):
  - Drives per-repo, per-skill chat interactions.
  - For `skill="qa"`:
    - Calls `RepoLibrary.query` to build a `query_plan`.
    - Validates that a QA adapter is present for the repo.
    - Delegates to `_execute_skill_chat` → `_format_qa_answer_stub`.
- `/api/qa_execute` (POST):
  - QA-focused convenience wrapper over `_execute_skill_chat` with
    a stable response shape:
    - `{"type": "qa_result", "status": "completed", "plan": ..., "answer": ...}`.

### Planning (`scripts/repo_library.RepoLibrary`)

- `RepoLibrary.query(question, mode=QueryMode.QA, repo_hint=..., qa_mode=...)`:
  - Loads `_manifest.json`.
  - Selects the target repo and computes `repo_context_keys`.
  - Uses `FileAdapterBank` to attach QA adapter metadata:
    - `plan["skills"][repo_id]["qa"] = adapter.info()`.

### QA Execution Core (`run.py`)

#### `_execute_skill_chat`

- Validates `skill`, `question`, `repo_hint`, `qa_mode`.
- For `skill == "qa"`:
  - Calls `repo_lib.query` to get a `query_plan`.
  - Ensures:
    - Exactly one repo (`repo_hint`) is present.
    - The QA adapter metadata exists under `plan["skills"][repo_hint]["qa"]`.
  - Calls `_format_qa_answer_stub(plan, qa_meta=qa_meta)`.
  - Wraps the result:

    ```json
    {
      "type": "skill_chat_result",
      "status": "completed",
      "skill": "qa",
      "plan": { ...query_plan... },
      "answer": "<final QA answer>"
    }
    ```

#### `_format_qa_answer_stub`

This function:

1. Opens the repo and graph:
   - `repo = open_repository(repo_id)`.
   - `graph = repo.graph`.
   - `entities = list(graph.entities())`, with `entities_by_id` map.
2. Performs retrieval:
   - If `qa_meta["index"]` is present:
     - Loads a `SimpleNumpyRepoIndex` via
       `modules.vector_index.load_simple_repo_index`.
     - Runs similarity search using:
       - The full natural-language question.
       - Code-like tokens extracted from the question (e.g.,
         identifiers like `update_screenshots`).
     - Aggregates hits by `entity_id` and resolves:
       - Artifact URIs via `graph.resolve` and `parse_program_uri`.
       - File paths and line spans.
     - Builds a `matches` list of candidate locations.
   - If no index hits:
     - Falls back to `graph.search_refs(tok)` on question tokens.
     - If still empty, runs a fuzzy name pass to handle small naming
       variations (e.g., pluralization).
3. Builds a `context_summary` string from `matches`:
   - Repository id, question, and a numbered list of:
     - `entity_kind`, `entity_name`, file path, line range, and matched
       token.
4. Builds a QA prompt:
   - Instructs the model to answer like a senior engineer.
   - Emphasizes short, chat-quality answers.
   - Includes the question and `context_summary` as reference.
5. Executes the QA model:
   - **Primary path (adapter-driven)**:
     - Uses `modules.qa_runtime.get_model_config_from_adapter(qa_meta)`
       to construct a `QAModelConfig`.
     - Loads/caches `(model, tokenizer)` via
       `modules.qa_runtime.get_or_load_model`.
     - Runs generation via
       `modules.qa_runtime.run_qa_generation(cfg, model, tokenizer, prompt)`.
   - **Fallback path**:
     - If adapter-driven runtime fails, falls back to the legacy global
       base LLM via `_llm_generate_answer`.
   - Returns the model answer, or `context_summary` on failure.

### QA Runtime (`modules/qa_runtime.py`)

- `QAModelConfig`:
  - Normalized configuration per adapter:
    - `model_name`, `model_id`, `cache_dir`.
    - `quantization` (prefers `"4bit"` by default).
    - `lora_path` (optional).
    - `max_new_tokens`, `temperature`, `top_p`.
    - `infer_devices` (defaults to `[0, 1]` for the two 3090 GPUs).
- `get_model_config_from_adapter(qa_meta)`:
  - Merges adapter metadata with `model.yml` defaults (via
    `modules.model_registry.get_model_config`).
  - Ensures a valid `model_id` and sensible generation defaults.
- `get_or_load_model(cfg)`:
  - Loads tokenizer and model:
    - Uses `cfg.model_id` and optional `cfg.model_path`.
    - If `cfg.quantization` is 4‑bit and bitsandbytes is available,
      applies a `BitsAndBytesConfig(load_in_4bit=True, ...)` and
      `device_map="auto"`.
    - Otherwise, loads in standard precision and moves the model to
      the first `infer_device` when CUDA is available.
  - Caches models keyed by `(model_id, quantization, lora_path)` so
    multiple repos can share weights.
- `run_qa_generation(cfg, model, tokenizer, prompt)`:
  - Tokenizes the prompt and sends it through `model.generate`.
  - Uses:
    - `cfg.max_new_tokens` for length.
    - `cfg.temperature` and `cfg.top_p` for sampling.
  - Returns the decoded answer string.

### GPU Usage

- **Training (future)**:
  - Intended to use GPUs 0 and 1 (the two 24GB 3090s) for QA adapter
    training tasks.
- **Inference (current)**:
  - QA adapters default to `infer_devices = [0, 1]`.
  - When 4‑bit is enabled and `device_map="auto"` is used, the model
    is sharded across the visible GPUs (usually the two 3090s).
  - GPU 2 (12GB 3060) can be incorporated later by adjusting
    `infer_devices` or environment-level device visibility when
    sequence lengths and workloads are small enough.


