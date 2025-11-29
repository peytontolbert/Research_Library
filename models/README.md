### Models Directory Overview

This `models/` directory is the home for **all model-building code** for your arXiv paper and repository AGI substrate. It is intended to be the **single, coherent place** where the 32 models described in `models.md` are implemented, configured, and wired up to the rest of the system.

- **Design doc**: See `models.md` in this directory for the master specification of the 32 models (tiers 1–7).
- **Scope**: Any **new logic, classes, configs, adapters, or training scripts that are specific to these 32 models must live inside `models/`**.
- **External modules**: You are encouraged to **reuse existing modules and scripts** elsewhere in the repo (e.g., data loaders, corpus utilities, arXiv ingestion tools), but treat them as **dependencies**, not as places to add model-specific code.

---

### Tier Structure (High-Level)

The models are grouped into tiers, as documented in `models.md`:

- **Tier 1 – Metadata Models**: Operate on arXiv metadata (titles, abstracts, authors, categories, citation graph). Examples: metadata embedding, category classifier, citation and link prediction, author/community embeddings.
- **Tier 2 – Abstract-Level Models**: Fast, abstract-only models for code relevance, paper method summaries, keyword prediction, and abstract → repo planning.
- **Tier 3 – Full-PDF Models**: Work over parsed PDFs (text, equations, tables, figures) for full-paper language modeling, algorithm extraction, equation reasoning, and figure/table interpretation.
- **Tier 4 – Repository Models**: Code-grounded models over ASTs and repo graphs: repo/file embeddings, code Q&A, bug localization, code mutation, repo-to-repo similarity.
- **Tier 5 – Cross-Domain Models**: Bridge paper ↔ code, including paper → code generation, repo–paper alignment, PCA/RCA adapters, adapter fusion, and cross-modal retrieval.
- **Tier 6 – Universal Reasoning Models**: Unified AGI-style models that fuse papers, code, and metadata for multi-objective reasoning, planning, and action selection.
- **Tier 7 – Self-Play & Skill Models**: Models for autonomous improvement (skill adapters, reward models, curriculum generation).

For precise definitions (inputs, targets, objectives), always defer to `models.md`.

---

### How to Organize Code Under `models/`

The goal is to keep implementation **modular and tier-aligned**, while ensuring all model-specific logic remains here.

- **Recommended layout (example)**:
  - `models/tier1_metadata/` – Metadata graph models, embeddings, and trainers.
  - `models/tier2_abstract/` – Abstract-level classifiers and summarizers.
  - `models/tier3_pdf/` – PDF tokenization adapters, full-paper LMs, equation/figure models.
  - `models/tier4_repo/` – Repo embeddings, bug localization, mutation, and similarity models.
  - `models/tier5_cross_domain/` – PCA/RCA adapters, paper → code generator, repo–paper alignment.
  - `models/tier6_unified/` – Unified knowledge model, planner/adapter, PAVU-style controller.
  - `models/tier7_self_play/` – Skill adapter generator, reward model, curriculum models.
  - `models/shared/` – Generic utilities shared across tiers (e.g., base modules, loss utilities, configs) that are still **model-specific** to this substrate.
  - `models/experiments/` – JSON specs per model (see `PLAN.md` schema) pointing at `/data/checkpoints` for cache/checkpoints and `meta-llama/Llama-3.2-1B` as the base model.
  - `models/cli.py` – Simple CLI scaffold to load an experiment config and construct the stubbed model via the registry (ready to be wired to HF/PEFT/bitsandbytes).

You do not need to adopt this exact layout, but **any new directory or module that implements one of the 32 models should be created under `models/`**.

---

### Interaction with Existing Code

You can and should **reuse existing scripts and modules** elsewhere in the repository, especially for:

- **Data access & preprocessing**:
  - arXiv metadata ingestion and storage
  - PDF download/organization and parsing
  - Repository graphs and program graphs
- **Infrastructure**:
  - GCS interaction
  - Caching and local storage
  - Logging and metrics

However, when you:

- define model architectures,
- implement training/evaluation loops,
- define model-specific configs/checkpoint logic,
- create adapters (PCA/RCA/skills),

place that code here in `models/`, then import and call external utilities as needed.

---

### Adding a New Model (Pattern)

When you start implementing one of the models from `models.md`, follow a consistent pattern:

1. **Create a tier-specific module**
   - Example: for the Metadata Embedding Model, create something like `models/tier1_metadata/metadata_embedding.py`.
2. **Define a clear interface**
   - E.g., `forward(input_batch) -> outputs`, or a trainer class with `fit`, `evaluate`, and `encode`/`predict` methods.
3. **Keep configs near the code**
   - Store model hyperparameters/configs in this folder (e.g., `configs/` or simple YAML/JSON next to the model), not scattered across scripts outside `models/`.
4. **Use existing data utilities as dependencies**
   - Import from `modules/` or `scripts/` to access data, but avoid pushing model logic back into those folders.
5. **Document assumptions**
   - Add short docstrings/comments that reference the corresponding model number/name from `models.md` (e.g., “Implements Model 15: Repo Embedding Model (RCA Base)”).

---

### Experiment Configs and CLI

- Per-model JSON specs live in `models/experiments/`. They currently pin:
  - `base_model`: `meta-llama/Llama-3.2-1B`
  - `adapter_type`: `lora`
  - `finetune_strategy`: `peft_lora`
  - `cache_dir` / `checkpoint_dir`: `/data/checkpoints`
- `models/cli.py` loads a config, builds a stub model from the registry, and (optionally) constructs HF tokenizer/backbone with PEFT/bitsandbytes if installed.
- Usage:
  - Dry run: `python -m models.cli --experiment models/experiments/m1_metadata_embedding.json --dry-run`
  - Train (stub): `python -m models.cli --experiment models/experiments/m1_metadata_embedding.json --mode train`
  - Make targets: `make -C models train.M1` or `make -C models eval.M1` (pattern based on experiment filenames).

---

### Long-Term Intent

The long-term goal is that this `models/` directory becomes a **self-contained substrate of trainable models**:

- Each of the **32 models** in `models.md` has a concrete implementation (or at least a scaffold) here.
- Training, evaluation, and inference entrypoints for these models are discoverable from here (or from scripts that live in this directory).
- Higher-level systems (servers, agents, orchestration code) import models from `models/` rather than duplicating model logic elsewhere.

As you build out the system, treat `models/` as the **authoritative home** for learning components, with other parts of the repository providing data, infrastructure, and orchestration support.
