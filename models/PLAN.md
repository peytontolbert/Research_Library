We treat your 32-model substrate as a single research system: a layered, adapter-driven world-model over metadata → abstracts → PDFs → repos → skills → self-play.

Below is a whitepaper-style specification with:

Diagrams showing how all 32 models connect

Training pipeline archetypes and per-model notes

JSON specs (schema + examples) for dataset → training → eval

Training order / dependency schedule

Dataset size & GPU cost estimates (ballpark, for planning)

1. Global Architecture: How All 32 Models Connect
1.1 Tier Overview

We keep your 32 models in 7 tiers (as we defined previously):

Metadata Models (1–5)

Abstract-Level Models (6–9)

PDF-Level Models (10–14)

Repository Models (15–20)

Cross-Domain Models (21–26)

Unified AGI Models (27–29)

Self-Play & Skill Models (30–32)

1.2 High-Level System Graph
flowchart TD
    subgraph T1[Tier 1: Metadata Models]
        M1[1. Metadata Embedding]
        M2[2. Category Classifier]
        M3[3. Citation Prediction]
        M4[4. Link Prediction (Graph)]
        M5[5. Author Embedding]
    end

    subgraph T2[Tier 2: Abstract-Level Models]
        A1[6. Abstract→Embedding/Relevance]
        A2[7. Abstract→Method Summary]
        A3[8. Abstract→Keywords]
        A4[9. Abstract→Repo Planning]
    end

    subgraph T3[Tier 3: PDF Models]
        P0[10. PDF Tokenization]
        P1[11. Full-Paper LM]
        P2[12. Section→Algorithm]
        P3[13. Equation Reasoning]
        P4[14. Figure/Table Interpretation]
    end

    subgraph T4[Tier 4: Repo Models]
        R1[15. Repo Embedding (RCA Base)]
        R2[16. File Embedding]
        R3[17. Code Q&A]
        R4[18. Bug Localization]
        R5[19. Code Mutation]
        R6[20. Repo-Repo Similarity]
    end

    subgraph T5[Tier 5: Cross-Domain Models]
        C1[21. Paper→Code Generator]
        C2[22. Repo-Paper Alignment]
        C3[23. Paper-Cond. Adapter (PCA)]
        C4[24. Repo-Cond. Adapter (RCA)]
        C5[25. Adapter Fusion]
        C6[26. Cross-Modal Retrieval]
    end

    subgraph T6[Tier 6: Unified AGI]
        U1[27. Unified Knowledge Model]
        U2[28. World/Planner Adapter]
        U3[29. Action Selector (PAVU)]
    end

    subgraph T7[Tier 7: Self-Play]
        S1[30. Skill Adapter Generator]
        S2[31. Self-Play Reward Model]
        S3[32. Curriculum Generation]
    end

    % Flows
    T1 --> T2
    T1 --> T3
    T1 --> T4

    T2 --> T5
    T3 --> T5
    T4 --> T5

    T5 --> T6
    T6 --> T7
    T7 --> T6

1.3 Dataflow Detail: From Raw Assets to AGI Loop
flowchart LR
    RawMeta[Raw Metadata (1.7M)] -->|preprocess| M1 & M2 & M3 & M4 & M5
    RawPDF[500GB PDFs] -->|OCR + structure| P0 --> P1 & P2 & P3 & P4
    Repos[Code Repositories] -->|AST + static analysis| R1 & R2 & R6

    M1 --> A1
    M2 & M3 & M4 & M5 --> A2 & A3 & A4

    A1 & A4 & R1 & R2 & R6 --> C2
    P1 & P2 & P3 & P4 --> C1

    C1 & C2 & (PCA=C3) & (RCA=C4) & C5 & C6 --> U1
    U1 --> U2 --> U3

    U3 --> S1 & S2 & S3
    S1 & S2 & S3 --> U1

2. Training Pipelines: Archetype + Per-Model Mapping

Instead of 32 totally distinct pipelines, we define 5 canonical pipeline archetypes and map each model to one of them, with notes.

2.1 Canonical Pipeline Archetypes

Archetype A – Embedding / Contrastive Model
Used by: M1, M5, A1, R1, R2, R6, C6

Stages:

Data ingestion → pairs or triplets ((x_i, x_j, label/similarity)).

Tokenization / feature extraction.

Encoder forward pass → embedding z.

Contrastive loss (InfoNCE / cosine similarity) or metric learning.

Eval via retrieval metrics (Recall@k, NDCG).

Archetype B – Classifier / Regressor
Used by: M2, M3, M4, A3, A4, R4, S2

Stages:

Labeled samples (x, y).

Backbone encoder (text / code / graph).

Classification head → logits / scores.

Cross-entropy / ranking loss.

Eval via accuracy, F1, AUC, MRR.

Archetype C – Generative Sequence / LLM Finetune
Used by: P1, P2, P3, P4, A2, C1, R3, R5, U1, U2, U3, S1, S3

Stages:

Input–output pairs (context, target_sequence).

Tokenization (text/code/math-aware).

Autoregressive training (next-token prediction / teacher forcing).

Eval via perplexity + task-specific metrics (BLEU, Pass@k, etc.).

Archetype D – Graph / GNN Model
Used by: M4, M5, possibly R6

Stages:

Build graph (nodes = papers/authors/repos, edges = citations/coauthor/related).

Node feature initialization (embeddings from other models).

GNN propagation steps.

Node/edge prediction loss.

Link prediction metrics (Hits@k, ROC AUC).

Archetype E – Policy / RL / Self-Play
Used by: R5, U3, S1, S2, S3

Stages:

Define environment (tasks, repos, simulators).

Policy network (often initialized from generative LLM).

Rollouts: plan→act→verify→reward→update.

RL objective (PPO/REINFORCE) or offline RL/BC.

Eval via downstream task success (pass@k, solved tasks count, etc.).

2.2 Model-to-Pipeline Mapping (with Short Specs)

Below is each model with:

Archetype

Input/Target reminder

Loss / Core Objective

Tier 1: Metadata Models

M1 – Metadata Embedding Model

Archetype: A (Embedding)

Input: (title, abstract, categories, authors)

Target: vector; training via pseudo-labels: co-citation, same-category.

Loss: contrastive (InfoNCE) on positive (same paper / co-cited) vs negatives.

M2 – Metadata Category Classifier

Archetype: B (Classifier)

Input: title + abstract

Target: arXiv category labels (multi-class / multi-label).

Loss: multi-label cross-entropy.

M3 – Citation Prediction Model

Archetype: B / D hybrid

Input: paper embedding + candidate citees

Target: edges to likely citations.

Loss: ranking loss / sampled softmax over papers.

M4 – Paper Similarity / Link Prediction

Archetype: D (Graph)

Input: nodes in citation/co-author graph

Target: missing edges

Loss: link prediction (binary cross-entropy over sampled edges).

M5 – Author Embedding / Community Model

Archetype: A + D

Input: author node and local graph neighborhood

Target: embedding; can use community detection or author labels as supervision.

Tier 2: Abstract-Level Models

A1 – Abstract→Repo Relevance / Embedding

Archetype: A (Embedding)

Input: abstract; candidate repo descriptors

Target: aligned embeddings (abstract close to relevant repos).

A2 – Abstract→Method Summary Model

Archetype: C (Generative)

Input: abstract

Target: compressed method summary.

A3 – Abstract→Keywords Model

Archetype: B / C

Input: abstract

Target: keywords; trained either as classification (vocab of tags) or generative (keyword sequence).

A4 – Abstract→Repo Planning Model

Archetype: B (Classifier)

Input: abstract + repo graph features

Target: modules/classes to be touched.

Loss: multi-label classification over repo modules.

Tier 3: PDF-Level Models

P0 – PDF Tokenization / Structuring

Archetype: custom preprocessing pipeline (no big model needed, but can include layout/vision model).

P1 – Full-Paper LM

Archetype: C (LLM finetune)

Input: full paper tokens

Target: next-token prediction.

P2 – Section→Algorithm Model

Archetype: C

Input: Method section chunk

Target: algorithm in pseudo-code.

P3 – Equation Reasoning Model

Archetype: C (text+math)

Input: equation + surrounding text

Target: explanation / next steps / simplification.

P4 – Figure/Table Interpretation Model

Archetype: C (multimodal if you want)

Input: image/table representation

Target: textual description or extracted values.

Tier 4: Repository Models

R1 – Repo Embedding (RCA Base)

Archetype: A

Input: repo content (files, README, structure)

Target: embedding; aligning repos by functional similarity.

R2 – File-Level Embedding Model

Archetype: A

Input: source file

Target: file-level embedding.

R3 – Code Q&A Model

Archetype: C (Generative)

Input: (code context, natural language question)

Target: natural language answer grounded in code.

R4 – Bug Localization Model

Archetype: B (Classifier)

Input: (test failure description, diff history, code context)

Target: distribution over files / lines.

R5 – Code Mutation Model

Archetype: C + E (RL fine-tune)

Input: code context + bug/test failure

Target: minimal patch; reward from tests.

R6 – Repo-to-Repo Similarity Model

Archetype: A / D

Input: pair of repo embeddings

Target: similarity score or classification of “related / unrelated.”

Tier 5: Cross-Domain Models

C1 – Paper→Code Generator

Archetype: C (LLM finetune)

Input: methods/pseudocode from paper

Target: runnable code (PyTorch, etc.).

C2 – Repo-Paper Alignment Model

Archetype: A/B

Input: (paper, repo) embeddings

Target: link prediction / alignment score.

C3 – Paper-Conditioned Adapter (PCA)

Archetype: meta-model over LoRA deltas

Input: paper embedding

Target: adapter weights (ΔW) for base model.

C4 – Repo-Conditioned Adapter (RCA)

Same as C3 but conditioned on repo embedding.

C5 – Adapter Fusion Model (PCA ⊕ RCA)

Archetype: small neural network mapping (ΔW_paper, ΔW_repo) → fused ΔW.

C6 – Cross-Modal Retrieval Model (PDF ↔ Repo)

Archetype: A (Embedding)

Input: paper chunk OR code snippet

Target: embedding in shared space.

Tier 6: Unified AGI Models

U1 – Unified Knowledge Model (Core AGI)

Archetype: C (LLM) with access to PCA/RCA + retrieval.

Input: multi-source context (papers, repos, metadata)

Target: unified reasoning outputs (answers, plans, code).

U2 – World/Planner Adapter

Archetype: C

Input: goal / objective + knowledge state

Target: plan (sequence of sub-goals & tool calls).

U3 – Action Selector (PAVU Loop)

Archetype: E (policy)

Input: system state (plan, progress, errors)

Target: {Plan, Act, Verify, Update, Stop} or more fine-grained actions.

Tier 7: Self-Play & Skill Models

S1 – Skill Adapter Generator

Archetype: C + E

Input: task distribution + performance traces

Target: new skill adapters (ΔW) specialized to tasks.

S2 – Self-Play Reward Model

Archetype: B / E

Input: (task spec, trajectory, outputs, tests)

Target: scalar reward / quality label.

S3 – Curriculum Generation Model

Archetype: C + E

Input: agent performance history

Target: new tasks of appropriate difficulty.

3. JSON Specs: Dataset → Training → Evaluation

We define a canonical experiment spec (JSON), then show concrete examples.

3.1 Canonical Experiment Schema
{
  "experiment_name": "string",
  "model_id": "string",           
  "tier": "T1_metadata | T2_abstract | T3_pdf | T4_repo | T5_cross | T6_unified | T7_selfplay",
  "backbone": {
    "type": "encoder | decoder | encoder_decoder | graph | policy",
    "base_model": "llama-3-8b | code-llama-7b | custom-gnn | etc.",
    "adapter_type": "none | lora | pca | rca | fusion",
    "parameters_millions": 8000
  },
  "dataset": {
    "sources": [
      "arxiv_metadata",
      "arxiv_pdfs",
      "github_repos",
      "generated_selfplay_traces"
    ],
    "filters": {
      "years": [2000, 2025],
      "categories": ["cs.LG", "cs.AI"],
      "languages": ["python", "cpp"]
    },
    "construction": {
      "input_fields": ["title", "abstract"],
      "target_fields": ["categories"],
      "max_samples": 1000000,
      "train_val_test_split": [0.8, 0.1, 0.1],
      "shuffling_seed": 42
    },
    "tokenization": {
      "tokenizer_name": "llama-3-tokenizer",
      "max_source_tokens": 2048,
      "max_target_tokens": 512,
      "truncate_strategy": "longest_first"
    }
  },
  "training": {
    "objective": "cross_entropy | contrastive | rl_ppo | link_prediction",
    "batch_size": 64,
    "gradient_accumulation_steps": 2,
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "precision": "bf16",
    "gradient_checkpointing": true
  },
  "evaluation": {
    "metrics": [
      "accuracy",
      "f1",
      "ndcg_at_10",
      "recall_at_50",
      "perplexity"
    ],
    "eval_interval_steps": 5000,
    "early_stopping": {
      "metric": "validation_loss",
      "patience": 3
    }
  },
  "inference": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "beam_size": 1
  },
  "compute": {
    "gpus": [
      {
        "type": "RTX_3090",
        "count": 4,
        "memory_gb": 24
      }
    ],
    "estimated_gpu_hours": 120,
    "distributed_strategy": "ddp"
  }
}

3.2 Example: M2 – Metadata Category Classifier
{
  "experiment_name": "M2_metadata_category_classifier_v1",
  "model_id": "M2",
  "tier": "T1_metadata",
  "backbone": {
    "type": "encoder",
    "base_model": "mini-llama-1b",
    "adapter_type": "lora",
    "parameters_millions": 1000
  },
  "dataset": {
    "sources": ["arxiv_metadata"],
    "filters": {
      "years": [1990, 2025]
    },
    "construction": {
      "input_fields": ["title", "abstract"],
      "target_fields": ["primary_category"],
      "max_samples": 1700000,
      "train_val_test_split": [0.8, 0.1, 0.1],
      "shuffling_seed": 42
    },
    "tokenization": {
      "tokenizer_name": "llama-3-tokenizer",
      "max_source_tokens": 512,
      "max_target_tokens": 32,
      "truncate_strategy": "longest_first"
    }
  },
  "training": {
    "objective": "cross_entropy",
    "batch_size": 256,
    "gradient_accumulation_steps": 1,
    "num_epochs": 3,
    "learning_rate": 3e-5,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "precision": "bf16",
    "gradient_checkpointing": true
  },
  "evaluation": {
    "metrics": ["accuracy", "macro_f1"],
    "eval_interval_steps": 2000,
    "early_stopping": {
      "metric": "macro_f1",
      "patience": 2
    }
  },
  "compute": {
    "gpus": [
      {
        "type": "RTX_3090",
        "count": 2,
        "memory_gb": 24
      }
    ],
    "estimated_gpu_hours": 30,
    "distributed_strategy": "ddp"
  }
}

3.3 Example: P1 – Full-Paper LM
{
  "experiment_name": "P1_full_paper_lm_v1",
  "model_id": "P1",
  "tier": "T3_pdf",
  "backbone": {
    "type": "decoder",
    "base_model": "llama-3-8b",
    "adapter_type": "lora",
    "parameters_millions": 8000
  },
  "dataset": {
    "sources": ["arxiv_pdfs_structured"],
    "filters": {
      "years": [2005, 2025],
      "categories": ["cs.LG", "cs.AI", "cs.CL"]
    },
    "construction": {
      "input_fields": ["full_paper_tokens"],
      "target_fields": ["full_paper_tokens_shifted"],
      "max_samples": 0,
      "train_val_test_split": [0.9, 0.05, 0.05],
      "shuffling_seed": 123
    },
    "tokenization": {
      "tokenizer_name": "llama-3-tokenizer",
      "max_source_tokens": 4096,
      "max_target_tokens": 4096,
      "truncate_strategy": "sliding_window"
    }
  },
  "training": {
    "objective": "cross_entropy",
    "batch_size": 8,
    "gradient_accumulation_steps": 8,
    "num_epochs": 1,
    "learning_rate": 1e-5,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "warmup_steps": 5000,
    "precision": "bf16",
    "gradient_checkpointing": true
  },
  "evaluation": {
    "metrics": ["perplexity"],
    "eval_interval_steps": 1000,
    "early_stopping": {
      "metric": "validation_loss",
      "patience": 3
    }
  },
  "compute": {
    "gpus": [
      {
        "type": "RTX_3090",
        "count": 4,
        "memory_gb": 24
      }
    ],
    "estimated_gpu_hours": 200,
    "distributed_strategy": "ddp"
  }
}

3.4 Example: C1 – Paper→Code Generator
{
  "experiment_name": "C1_paper_to_code_v1",
  "model_id": "C1",
  "tier": "T5_cross",
  "backbone": {
    "type": "decoder",
    "base_model": "code-llama-13b",
    "adapter_type": "pca",
    "parameters_millions": 13000
  },
  "dataset": {
    "sources": ["arxiv_pdfs_structured", "github_repos"],
    "filters": {
      "categories": ["cs.LG", "cs.AI"],
      "languages": ["python"]
    },
    "construction": {
      "input_fields": ["method_section_text"],
      "target_fields": ["code_snippet"],
      "max_samples": 500000,
      "train_val_test_split": [0.85, 0.1, 0.05],
      "shuffling_seed": 77
    },
    "tokenization": {
      "tokenizer_name": "code-llama-tokenizer",
      "max_source_tokens": 2048,
      "max_target_tokens": 2048,
      "truncate_strategy": "longest_first"
    }
  },
  "training": {
    "objective": "cross_entropy",
    "batch_size": 4,
    "gradient_accumulation_steps": 16,
    "num_epochs": 2,
    "learning_rate": 1e-5,
    "optimizer": "adamw",
    "weight_decay": 0.0,
    "warmup_steps": 2000,
    "precision": "bf16",
    "gradient_checkpointing": true
  },
  "evaluation": {
    "metrics": ["perplexity", "pass_at_1", "pass_at_5"],
    "eval_interval_steps": 1000,
    "early_stopping": {
      "metric": "pass_at_1",
      "patience": 3
    }
  },
  "compute": {
    "gpus": [
      {
        "type": "RTX_3090",
        "count": 4,
        "memory_gb": 24
      }
    ],
    "estimated_gpu_hours": 250,
    "distributed_strategy": "ddp"
  }
}


You can instantiate similar specs for all 32 models by filling:

model_id

tier

backbone.base_model

dataset input_fields / target_fields

training objective and metrics.

4. Training Order: Which Models to Train First

We want a dependency-respecting curriculum that also front-loads models giving max immediate utility.

Phase 0 – Infrastructure

PDF ingestion + structuring (P0)

Repo ingestion + static analysis (AST, call graphs)

Phase 1 – Metadata Foundation (Cheap, High-Leverage)

M1 – Metadata Embedding

M2 – Category Classifier

M4 – Paper Similarity / Link Prediction

(M3, M5 are nice but can come slightly later.)

Outcome: a global science graph and semantic embedding space over 1.7M papers.

Phase 2 – Repo Embeddings + Basic Code Q&A

R2 – File-Level Embedding

R1 – Repo Embedding (RCA Base)

R3 – Code Q&A

Outcome: you can query codebases and map papers → repos roughly.

Phase 3 – Abstract-Level Cross-Over

A1 – Abstract→Repo Relevance

A2 – Abstract→Method Summary

A3 – Keywords

Outcome: fast, high-level model of “what is this paper” + “which repos matter.”

Phase 4 – Full-Paper Modeling

P1 – Full-Paper LM

P2 – Section→Algorithm

P3 – Equation Reasoning

Outcome: deep scientific reasoning with math + algorithms.

Phase 5 – Cross-Domain Paper↔Code

C2 – Repo-Paper Alignment

C6 – Cross-Modal Retrieval (PDF ↔ Repo)

C1 – Paper→Code Generator

Outcome: your system starts to implement papers in code and align them to repos.

Phase 6 – Adapters (PCA/RCA) and Fusion

C3 – PCA

C4 – RCA

C5 – Adapter Fusion

Outcome: your AGI substrate can reconfigure itself per paper and per repo.

Phase 7 – Unified AGI and Self-Play

U1 – Unified Knowledge Model

U2 – World/Planner Adapter

U3 – Action Selector (PAVU)

R5 – Code Mutation (RL fine-tune)

S1 – Skill Adapter Generator

S2 – Reward Model

S3 – Curriculum Generation

Outcome: a self-play, self-improving scientific coding AGI grounded in your entire library.

5. Dataset Sizes & GPU Cost Estimates (Ballpark)

Assumptions:

You subsample heavily from 500GB PDFs (you don’t need to train on literally everything at first).

You use LoRA/PCA/RCA adapters, not full finetunes.

Hardware: 4× RTX 3090 (24GB) or similar.

These are approximate orders of magnitude:

Tier 1 – Metadata

M1/M2/M4:

Data: up to 1.7M examples; ~2–4 tokens per abstract (≈1–2B tokens).

Model: 1–3B encoder.

Cost: 20–60 GPU-hours each on 2×3090.

Tier 2 – Abstract-Level

A1/A2/A3/A4:

Data: 0.5–1.7M examples.

Token count: similar to Tier 1, maybe ~1B tokens total.

Cost: ~30–80 GPU-hours across these, especially for generative A2.

Tier 3 – Full-Paper

P1 with 200k full papers at 4k tokens ≈ 800M tokens:

8B model with LoRA, 4×3090, 1 epoch: 150–250 GPU-hours.

P2/P3 use subsets (e.g. 100k method/equation chunks); cost: 50–100 GPU-hours combined.

Tier 4 – Repos

Assume ~10–50 large repos (CPython, Linux, PyTorch, Transformers, etc.).

Tokens: ~200–500M tokens of code for initial phase.

R2/R1 (embeddings): 20–40 GPU-hours.

R3 (code Q&A): ~300M code tokens + QA pairs: 80–150 GPU-hours.

Tier 5 – Cross-Domain

C1 – Paper→Code: 500k paper-method↔code pairs, average ~2–3k source+target tokens.

~1–1.5B tokens effective.

LoRA on 13B code model: 200–300 GPU-hours.

C2/C6: embedding alignment tasks: 20–60 GPU-hours.

C3/C4/C5: adapter training is relatively cheap: 20–50 GPU-hours total.

Tier 6 – Unified AGI

U1 as an 8–13B LLM with adapters, trained on mixtures of all previous corpora (metadata, abstracts, code, PDFs) but heavily sampled (e.g. 1–2B tokens):

Cost: 250–400 GPU-hours.

U2/U3: built on top as smaller finetunes / RL: 50–150 GPU-hours combined.

Tier 7 – Self-Play

Costs are dominated by rollouts + code execution rather than pure training.

Expect ongoing compute, but you can begin with 50–100 GPU-hours to bootstrap S1/S2/S3.

6. Putting It Together

You now have:

A 32-model catalog with clear inputs/targets.

A connection graph across all tiers.

Canonical JSON specs for dataset → training → eval.

A phased curriculum for training.

Token + GPU estimates to plan around your 3090s.

If you want the next layer, I can:

Generate per-model JSON skeletons (M1…S3) as files.

Draft a Makefile / CLI spec like make train.M2, make eval.C1.

Or design the orchestration agent that reads these JSON specs and automatically:

builds datasets,

schedules training jobs,

runs evals,

logs everything into a persistent experiment registry.
