#!/usr/bin/env python3
"""Generate enhanced Hugging Face model cards for repository_library models."""

from __future__ import annotations

import json
from pathlib import Path

MIRROR_ROOT = Path("/arxiv/models/repository_library")
EXPERIMENT_ROOT = Path("/data/repository_library/models/experiments")
GITHUB_REPO = "https://github.com/peytontolbert/research_library"
HF_COLLECTION = "https://huggingface.co/collections/PeytonT/research-library-6a49c589ef4d763f7539b50d"

SLUG_TO_EXPERIMENT = {
    "abstract-code-relevance": "a1_abstract_code_relevance.json",
    "abstract-code-relevance-pairs": "a1_abstract_code_relevance.json",
    "abstract-method-summary": "a2_abstract_method_summary.json",
    "abstract-keywords": "a3_abstract_keywords.json",
    "abstract-repo-planning": "a4_abstract_repo_planning.json",
    "paper-to-code": "c1_paper_to_code.json",
    "repo-paper-alignment": "c2_repo_paper_alignment.json",
    "paper-conditioned-adapter": "c3_paper_conditioned_adapter.json",
    "repo-conditioned-adapter": "c4_repo_conditioned_adapter.json",
    "adapter-fusion": "c5_adapter_fusion.json",
    "cross-modal-retrieval": "c6_cross_modal_retrieval.json",
    "query-rewriter": "l1_query_rewriter.json",
    "cross-encoder-reranker": "l2_cross_encoder_reranker.json",
    "repo-state-grounding": "l3_repo_state_grounding.json",
    "candidate-row-reranker": "l4_candidate_row_reranker.json",
    "span-infill-gate": "l5_span_infill_gate.json",
    "graph-neighborhood-router": "l6_graph_neighborhood_router.json",
    "cluster-router": "l7_cluster_router.json",
    "heap-scheduler": "l8_heap_scheduler.json",
    "verifier-accept-policy": "l9_verifier_accept_policy.json",
    "jepa-repo-state": "l10_jepa_repo_state.json",
    "metadata-embedding": "m1_metadata_embedding.json",
    "metadata-category-classifier": "m2_metadata_category_classifier.json",
    "citation-prediction": "m3_citation_prediction.json",
    "paper-link-prediction": "m4_paper_link_prediction.json",
    "author-embedding": "m5_author_embedding.json",
    "paper-fulltext-embedding": "m6_paper_fulltext_embedding.json",
    "paper-sentence-embedding": "m7_paper_sentence_embedding.json",
    "pdf-tokenization": "p0_pdf_tokenization.json",
    "full-paper-lm": "p1_full_paper_lm.json",
    "section-to-algorithm": "p2_section_to_algorithm.json",
    "equation-reasoning": "p3_equation_reasoning.json",
    "figure-table-interpretation": "p4_figure_table_interpretation.json",
    "paper-qa": "p5_paper_qa.json",
    "repo-embedding": "r1_repo_embedding.json",
    "file-embedding": "r2_file_embedding.json",
    "bug-localization": "r4_bug_localization.json",
    "repo-similarity": "r6_repo_similarity.json",
    "self-play-reward-model": "s2_self_play_reward_model.json",
    "unified-knowledge-model": "u1_unified_knowledge_model.json",
    "world-planner-adapter": "u2_world_planner_adapter.json",
}

TASK_SUMMARIES = {
    "abstract-code-relevance": "Scores whether a paper abstract is relevant to a repository or code task.",
    "abstract-code-relevance-pairs": "Scores abstract and code pairs using the paired-training variant of the abstract-code-relevance model.",
    "abstract-method-summary": "Generates concise method summaries from scientific abstracts.",
    "abstract-keywords": "Generates normalized keyword strings from scientific abstracts.",
    "abstract-repo-planning": "Predicts repository-oriented implementation plans from abstract-level paper descriptions.",
    "paper-to-code": "Generates code-oriented outputs conditioned on paper content.",
    "repo-paper-alignment": "Aligns repository content with scientific paper content in a shared training setup.",
    "paper-conditioned-adapter": "Adapts the shared backbone toward paper-conditioned reasoning and generation tasks.",
    "repo-conditioned-adapter": "Adapts the shared backbone toward repository-conditioned reasoning and retrieval tasks.",
    "adapter-fusion": "Fuses paper-conditioned and repo-conditioned adapters into a shared cross-domain component.",
    "cross-modal-retrieval": "Retrieves semantically related items across paper and repository modalities.",
    "query-rewriter": "Rewrites search queries for better retrieval recall over repositories and paper assets.",
    "cross-encoder-reranker": "Reranks retrieved candidates with a cross-encoder scoring pass.",
    "repo-state-grounding": "Grounds retrieval and planning decisions in repository structure and symbol state.",
    "candidate-row-reranker": "Scores candidate evidence rows for acceptance or rejection inside the search stack.",
    "span-infill-gate": "Chooses between copy, infill, and regenerate behaviors for retrieved spans.",
    "graph-neighborhood-router": "Routes retrieval expansion over local repository graph neighborhoods.",
    "cluster-router": "Routes tasks into the best repository, paper, or concept cluster before expensive reasoning.",
    "heap-scheduler": "Schedules unresolved candidates, hard negatives, and verification tasks in the search stack.",
    "verifier-accept-policy": "Predicts whether retrieved or generated candidates should be accepted, repaired, or broadened.",
    "jepa-repo-state": "Serves as the current repository-state representation learner and JEPA-style proxy experiment.",
    "metadata-embedding": "Builds dense embeddings over paper metadata fields such as title, abstract, authors, and categories.",
    "metadata-category-classifier": "Predicts arXiv-style subject categories from metadata text.",
    "citation-prediction": "Predicts likely citation links between papers.",
    "paper-link-prediction": "Predicts paper-to-paper similarity or graph links.",
    "author-embedding": "Learns author and community representations from paper metadata and graph context.",
    "paper-fulltext-embedding": "Produces embeddings over full paper text for retrieval and clustering tasks.",
    "paper-sentence-embedding": "Produces sentence-level embeddings over paper text.",
    "pdf-tokenization": "Normalizes and models structured PDF-derived paper content for downstream paper tasks.",
    "full-paper-lm": "Models full-paper text for downstream scientific generation and conditioning tasks.",
    "section-to-algorithm": "Generates algorithmic descriptions from method or section-level paper text.",
    "equation-reasoning": "Models paper equations and mathematical reasoning spans.",
    "figure-table-interpretation": "Interprets figure and table content from structured paper representations.",
    "paper-qa": "Answers questions over paper text and structured paper context.",
    "repo-embedding": "Produces repository-level dense representations for retrieval and alignment.",
    "file-embedding": "Produces file-level dense representations for retrieval and matching.",
    "bug-localization": "Ranks likely bug locations inside repository content.",
    "repo-similarity": "Scores repository-to-repository similarity.",
    "self-play-reward-model": "Scores self-play traces and policy outputs for downstream training loops.",
    "unified-knowledge-model": "Combines paper, repository, and metadata inputs into a shared reasoning model.",
    "world-planner-adapter": "Adds planning behavior on top of the unified model for multi-step repository and paper tasks.",
}

PIPELINE_BY_MODEL_TYPE = {
    "seq2seq": "text2text-generation",
    "causal_lm": "text-generation",
    "classifier": "text-classification",
    "embedding": "feature-extraction",
}

DATASET_SOURCE_TEXT = {
    "arxiv_metadata": "arXiv metadata records spanning titles, abstracts, authors, and category labels.",
    "paper_text_parquet": "full-text paper corpus records prepared for model training.",
    "arxiv_pdfs_structured": "structured PDF shards containing text, equations, figures, and tables.",
    "arxiv_pdfs": "raw arXiv PDF documents routed through the Repository Library paper pipeline.",
    "github_repos": "repository graph and code chunk data exported from the Repository Library repo pipeline.",
    "paper_repo_pairs": "paper-to-repository alignment pairs built from the local alignment pipeline.",
    "paper_universe_graph": "paper graph features derived from the Repository Library paper-universe graph.",
    "generated_selfplay_traces": "generated self-play traces collected for reward-model supervision.",
}

ROLE_TEXT = {
    "T1_metadata": "metadata-layer component in the paper understanding stack",
    "T2_abstract": "abstract-layer component for fast paper interpretation",
    "T3_pdf": "full-paper or structured-PDF component",
    "T6_unified": "cross-domain reasoning component",
    "repository_library_search_stack": "search-stack component for retrieval, reranking, or routing",
}


def prettify_slug(slug: str) -> str:
    text = slug.replace("-", " ").title()
    replacements = {"Lm": "LM", "Qa": "QA", "Pdf": "PDF", "Jepa": "JEPA"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def infer_artifact_type(model_dir: Path) -> tuple[str, str]:
    if (model_dir / "adapter_model.safetensors").exists():
        return "peft", "LoRA adapter"
    return "transformers", "full fine-tuned model"


def infer_model_type(cfg: dict) -> str:
    training = cfg.get("training", {})
    if training.get("model_type"):
        return training["model_type"]
    backbone_type = cfg.get("backbone", {}).get("type", "")
    if backbone_type == "encoder_decoder":
        return "seq2seq"
    if backbone_type == "encoder":
        return "classifier"
    return "embedding"


def dataset_items(cfg: dict) -> list[str]:
    ds = cfg.get("dataset", {})
    construction = ds.get("construction", {})
    items = []
    dataset_id = construction.get("paper_dataset_id")
    if dataset_id:
        items.append(f"Primary published dataset: `{dataset_id}`")
    for source in ds.get("sources", []):
        desc = DATASET_SOURCE_TEXT.get(source, source.replace("_", " "))
        items.append(f"Source `{source}`: {desc}")
    return items


def task_tags(slug: str, cfg: dict) -> list[str]:
    tags = ["repository-library", "research-library", slug]
    model_id = cfg.get("model_id")
    tier = cfg.get("tier")
    if model_id:
        tags.append(model_id.lower())
    if tier:
        tags.append(str(tier).lower())
    if "paper" in slug:
        tags.extend(["papers", "scientific-papers"])
    if "repo" in slug or "file" in slug or "bug" in slug:
        tags.extend(["repositories", "code"])
    if "embedding" in slug:
        tags.append("embeddings")
    if any(token in slug for token in ["retrieval", "reranker", "router"]):
        tags.append("retrieval")
    if "qa" in slug:
        tags.append("question-answering")
    if "planning" in slug or "planner" in slug:
        tags.append("planning")
    return sorted(dict.fromkeys(tags))


def usage_snippet(repo_id: str, base_model: str, artifact_type: str, model_type: str) -> str:
    if artifact_type == "LoRA adapter":
        if model_type == "classifier":
            model_cls = "AutoModelForSequenceClassification"
        elif model_type == "seq2seq":
            model_cls = "AutoModelForSeq2SeqLM"
        else:
            model_cls = "AutoModel"
        return f"""```python
from transformers import {model_cls}, AutoTokenizer
from peft import PeftModel

repo_id = \"{repo_id}\"
base_id = \"{base_model}\"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
base = {model_cls}.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, repo_id)
```"""
    if model_type == "classifier":
        model_cls = "AutoModelForSequenceClassification"
    elif model_type == "seq2seq":
        model_cls = "AutoModelForSeq2SeqLM"
    else:
        model_cls = "AutoModel"
    return f"""```python
from transformers import {model_cls}, AutoTokenizer

repo_id = \"{repo_id}\"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = {model_cls}.from_pretrained(repo_id)
```"""


def render_card(slug: str, cfg: dict) -> str:
    model_dir = MIRROR_ROOT / slug
    library_name, artifact_type = infer_artifact_type(model_dir)
    model_type = infer_model_type(cfg)
    pipeline_tag = PIPELINE_BY_MODEL_TYPE.get(model_type, "feature-extraction")
    base_model = cfg.get("backbone", {}).get("base_model", "unknown")
    title = prettify_slug(slug)
    summary = TASK_SUMMARIES.get(slug, f"{title} for the Repository Library research stack.")
    repo_id = f"PeytonT/{slug}"
    ds_items = dataset_items(cfg)
    training = cfg.get("training", {})
    tokenization = cfg.get("dataset", {}).get("tokenization", {})
    construction = cfg.get("dataset", {}).get("construction", {})
    evaluation = cfg.get("evaluation", {})
    notes = cfg.get("notes")
    compute = cfg.get("compute", {})
    experiment_name = SLUG_TO_EXPERIMENT[slug]
    tier = cfg.get("tier", "unknown")
    tier_role = ROLE_TEXT.get(tier, "specialized Repository Library component")

    frontmatter = [
        "---",
        f"base_model: {base_model}",
        f"library_name: {library_name}",
        f"pipeline_tag: {pipeline_tag}",
        "tags:",
    ]
    for tag in task_tags(slug, cfg):
        frontmatter.append(f"- {tag}")
    frontmatter.append("---")

    sources = [
        f"- Hugging Face repo: `https://huggingface.co/{repo_id}`",
        f"- Hugging Face collection: `{HF_COLLECTION}`",
        f"- GitHub repository: `{GITHUB_REPO}`",
        f"- Experiment config: `{GITHUB_REPO}/blob/main/models/experiments/{experiment_name}`",
        f"- Models directory: `{GITHUB_REPO}/tree/main/models`",
    ]

    model_details = [
        f"- Artifact type: {artifact_type}",
        f"- Base model: `{base_model}`",
        f"- Backbone type: `{cfg.get('backbone', {}).get('type', 'unknown')}`",
        f"- Model ID: `{cfg.get('model_id', 'unknown')}`",
        f"- Tier: `{tier}`",
        f"- Role in stack: {tier_role}",
    ]

    procedure = [
        f"- Sources: `{', '.join(cfg.get('dataset', {}).get('sources', [])) or 'unknown'}`",
        f"- Input fields: `{', '.join(construction.get('input_fields', [])) or 'unknown'}`",
        f"- Target fields: `{', '.join(construction.get('target_fields', [])) or 'unknown'}`",
        f"- Train/val/test split: `{construction.get('train_val_test_split', 'unknown')}`",
        f"- Max samples: `{construction.get('max_samples', 'unknown')}`",
        f"- Batch size: `{training.get('batch_size', 'unknown')}`",
        f"- Precision: `{training.get('precision', 'unknown')}`",
        f"- Objective: `{training.get('objective', 'unknown')}`",
        f"- Learning rate: `{training.get('learning_rate', 'unknown')}`",
        f"- Max source tokens: `{tokenization.get('max_source_tokens', 'unknown')}`",
        f"- Max target tokens: `{tokenization.get('max_target_tokens', 'unknown')}`",
        f"- Fine-tune strategy: `{training.get('finetune_strategy', 'unknown')}`",
    ]
    if training.get("max_steps") is not None:
        procedure.append(f"- Max steps: `{training.get('max_steps')}`")
    if construction.get("streaming") is not None:
        procedure.append(f"- Streaming dataset build: `{construction.get('streaming')}`")
    if notes:
        procedure.append(f"- Notes: {notes}")

    gpu_items = compute.get("gpus") or []
    if gpu_items:
        gpu_text = "; ".join(f"{g.get('count', '?')}x {g.get('type', 'GPU')} ({g.get('memory_gb', '?')} GB)" for g in gpu_items)
    else:
        gpu_text = "not specified"
    compute_lines = [
        f"- Hardware: {gpu_text}",
        f"- Distributed strategy: `{compute.get('distributed_strategy', 'unknown')}`",
        f"- Estimated GPU hours in config: `{compute.get('estimated_gpu_hours', 'unknown')}`",
    ]

    metrics = evaluation.get("metrics") or []
    eval_line = ", ".join(metrics) if metrics else "not specified"

    out = []
    out.extend(frontmatter)
    out.append("")
    out.append(f"# {title}")
    out.append("")
    out.append(summary)
    out.append("")
    out.append("## Model Details")
    out.append("")
    out.extend(model_details)
    out.append("")
    out.append("This model is part of the Repository Library stack, a research system for indexing, retrieving, aligning, and reasoning over scientific papers, structured paper content, repositories, and cross-domain links between them.")
    out.append("")
    out.append("## Model Sources")
    out.append("")
    out.extend(sources)
    out.append("")
    out.append("## Intended Use")
    out.append("")
    out.append(f"- Primary use: {summary}")
    out.append("- Downstream use: retrieval, ranking, planning, paper understanding, or cross-domain reasoning inside the broader Repository Library system, depending on the model family.")
    out.append("- Out of scope: production safety claims, benchmark claims beyond the tracked experiment config, or deployment without task-specific validation.")
    out.append("")
    out.append("## Training Data")
    out.append("")
    if ds_items:
        out.append("The training inputs for this package were assembled from the following Repository Library data sources:")
        out.append("")
        out.extend(f"- {item}" for item in ds_items)
    else:
        out.append("Training data details were not fully declared in the packaged experiment config.")
    out.append("")
    out.append("## Training Procedure")
    out.append("")
    out.extend(procedure)
    out.append("")
    out.append("## Compute")
    out.append("")
    out.extend(compute_lines)
    out.append("")
    out.append("## Evaluation")
    out.append("")
    out.append(f"- Declared metrics: `{eval_line}`")
    out.append("- Status: this card reflects the current tracked experiment configuration and packaged weights in the Repository Library model stack.")
    out.append("")
    out.append("## Usage")
    out.append("")
    out.append(usage_snippet(repo_id, base_model, artifact_type, model_type))
    out.append("")
    out.append("## Limitations")
    out.append("")
    out.append("- These cards are generated from tracked experiment metadata and packaged artifacts, not from a separate benchmark report or external audit.")
    out.append("- Several training sources are pipeline outputs from the Repository Library codebase rather than standalone public datasets.")
    out.append("- These models are components of a larger research system and should be validated in their target workflow before deployment.")
    out.append("")
    out.append("## Project Context")
    out.append("")
    out.append(f"- GitHub repository: `{GITHUB_REPO}`")
    out.append(f"- Model collection: `{HF_COLLECTION}`")
    out.append("- Publisher: `PeytonT`")
    out.append("")
    return "\n".join(out)


def main() -> None:
    for slug, exp_name in SLUG_TO_EXPERIMENT.items():
        model_dir = MIRROR_ROOT / slug
        if not model_dir.is_dir():
            continue
        cfg = load_json(EXPERIMENT_ROOT / exp_name)
        readme = model_dir / "README.md"
        readme.write_text(render_card(slug, cfg))
        print(f"updated {readme}")


if __name__ == "__main__":
    main()
