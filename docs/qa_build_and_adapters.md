## QA Build and Adapters

### Exports and Manifest

- Graph exports live under `exports/{repo_id}/`:
  - `{repo_id}.entities.jsonl`, `{repo_id}.edges.jsonl`,
    `{repo_id}.artifacts.jsonl`.
- `_manifest.json` at `exports/_manifest.json` tracks:
  - `repos[repo_id].repo_state`
  - `repos[repo_id].skills`
  - `repos[repo_id].indices`

These are consumed by `scripts.repo_library.RepoLibrary` and
`scripts.skill_build`.

### QA Skill Build (`scripts/skill_build.py`)

- Entry point: `build_skill(repo_id: str, skill: str, ...)`.
- For `skill == "qa"`:
  - Loads the repository via `open_repository`.
  - Builds a QA vector index with
    `modules.vector_index.build_repo_qa_index`.
    - Enumerates entities.
    - Builds simple texts `"<kind> <name>"`.
    - Embeds via `modules.embeddings.embed_texts`.
    - Saves embeddings/items under
      `exports/{repo_id}/indices/qa/`.
  - Constructs an `adapter_info` record:
    - Core fields:
      - `adapter_id = f"qa:{repo_id}"`
      - `repo_id`, `skill = "qa"`
      - `built_at`, `skill_schema_version`
    - Retrieval metadata:
      - `index` (SimpleNumpyRepoIndex meta) when index build succeeds.
    - **Model configuration** (new):
      - `model_name` (defaults to `"llama"`, backed by `model.yml`).
      - `model_id`, `cache_dir` resolved via
        `modules.model_registry.get_model_config("llama")`.
      - `quantization` (defaults to `"4bit"`).
      - `lora_path` (None by default; filled by future training).
      - Generation defaults:
        - `max_new_tokens` (256)
        - `temperature` (0.1)
        - `top_p` (0.95)
      - `infer_devices` (defaults to `[0, 1]` to prefer the two 3090s).
  - Registers the adapter in the registry via
    `scripts.registry.register_adapter`, writing to
    `exports/_adapters/adapter_registry.json`.
  - Updates `_manifest.json`:
    - `repos[repo_id].skills["qa"]` gets:
      - `status`, `skill_schema_version`, `last_built_at`,
        `adapter_id`, `repo_state`.
    - `repos[repo_id].indices["qa"]` gets the QA index metadata.

### Model Registry (`model.yml` and `modules/model_registry.py`)

- `model.yml` defines logical model names and their HF ids/paths:
  - `llama` → `meta-llama/Llama-3.1-8B-Instruct`.
  - `T5` → `google/t5-v1_1-base`.
  - `bert` → `google/bert-base-uncased`.
- `modules.model_registry` provides:
  - `ModelConfig` dataclass.
  - `get_model_config(name)` to look up entries (used by
    `build_skill` when constructing QA adapters).


