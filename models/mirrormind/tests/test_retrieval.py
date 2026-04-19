from pathlib import Path
import json

from models.mirrormind.retrieval import CoarseLaneRetriever


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_coarse_lane_retriever_fuses_repo_paper_and_bridge(tmp_path: Path):
    repo_semantic = tmp_path / "semantic_from_chunks.jsonl"
    align = tmp_path / "paper_repo_align.jsonl"
    spans = tmp_path / "paper_repo_span_align.jsonl"

    _write_jsonl(
        repo_semantic,
        [
            {
                "id": "OneFormer:repo_chunks",
                "entity_id": "OneFormer",
                "time_window": "",
                "scope": "repo_chunks",
                "summary_text": "Repo: OneFormer. Universal image segmentation built with transformer backbones and detectron2.",
                "key_concepts": ["universal", "segmentation", "transformer", "detectron2"],
                "dense": [1.0],
            },
            {
                "id": "OpenWebVoyager:repo_chunks",
                "entity_id": "OpenWebVoyager",
                "time_window": "",
                "scope": "repo_chunks",
                "summary_text": "Repo: OpenWebVoyager. Multimodal web agent stack for browsing and exploration.",
                "key_concepts": ["web", "agent", "multimodal", "exploration"],
                "dense": [1.0],
            },
        ],
    )

    _write_jsonl(
        align,
        [
            {
                "paper_id": "2211.06220",
                "paper_title": "OneFormer: One Transformer to Rule Universal Image Segmentation",
                "paper_abstract": "",
                "pdf_path": "/arxiv/pdfs/2211/2211.06220.pdf",
                "paper_text": "One transformer for universal image segmentation across semantic, instance, and panoptic tasks.",
                "repo_id": "OneFormer",
                "repo_path": "/data/repositories/OneFormer/oneformer/model.py",
                "repo_offset": 0,
                "repo_text": "detectron2 based segmentation model",
                "label": 1,
                "score": 0.6,
                "shared_terms": ["transformer", "segmentation"],
                "candidate_rank": 1,
                "negative_type": None,
            },
            {
                "paper_id": "2401.13919",
                "paper_title": "OpenWebVoyager: Building Multimodal Web Agents",
                "paper_abstract": "",
                "pdf_path": "/arxiv/pdfs/2401/2401.13919.pdf",
                "paper_text": "OpenWebVoyager studies multimodal web agents with iterative exploration.",
                "repo_id": "OpenWebVoyager",
                "repo_path": "/data/repositories/OpenWebVoyager/WebVoyager/agent.py",
                "repo_offset": 0,
                "repo_text": "web agent runtime",
                "label": 1,
                "score": 0.6,
                "shared_terms": ["web", "agent"],
                "candidate_rank": 1,
                "negative_type": None,
            },
        ],
    )

    _write_jsonl(
        spans,
        [
            {
                "paper_id": "2211.06220",
                "paper_title": "OneFormer: One Transformer to Rule Universal Image Segmentation",
                "pdf_path": "/arxiv/pdfs/2211/2211.06220.pdf",
                "repo_id": "OneFormer",
                "repo_path": "/data/repositories/OneFormer/oneformer/model.py",
                "repo_offset": 128,
                "paragraph_id": 4,
                "page_start": 2,
                "page_end": 2,
                "line_start": 18,
                "line_end": 22,
                "paper_text": "The model unifies universal image segmentation with a transformer decoder.",
                "repo_text": "class OneFormerModel: pass",
                "label": 1,
                "score": 0.52,
                "shared_terms": ["universal", "segmentation", "transformer"],
                "negative_type": None,
            },
            {
                "paper_id": "2401.13919",
                "paper_title": "OpenWebVoyager: Building Multimodal Web Agents",
                "pdf_path": "/arxiv/pdfs/2401/2401.13919.pdf",
                "repo_id": "OpenWebVoyager",
                "repo_path": "/data/repositories/OpenWebVoyager/WebVoyager/agent.py",
                "repo_offset": 64,
                "paragraph_id": 2,
                "page_start": 1,
                "page_end": 1,
                "line_start": 10,
                "line_end": 15,
                "paper_text": "Our multimodal web agent performs iterative exploration over real websites.",
                "repo_text": "def run_agent(): pass",
                "label": 1,
                "score": 0.48,
                "shared_terms": ["multimodal", "web", "agent", "exploration"],
                "negative_type": None,
            },
        ],
    )

    retriever = CoarseLaneRetriever(
        repo_semantic_path=str(repo_semantic),
        paper_repo_align_path=str(align),
        paper_repo_span_path=str(spans),
    )
    result = retriever.retrieve("universal image segmentation transformer", top_k_repos=2, top_k_papers=2, top_k_spans=2)

    assert result["fused_repos"][0]["repo_id"] == "OneFormer"
    assert result["paper_hits"][0]["paper_id"] == "2211.06220"
    assert result["support_spans"][0]["repo_id"] == "OneFormer"


def test_coarse_lane_retriever_paper_lane_can_bridge_repo_choice(tmp_path: Path):
    repo_semantic = tmp_path / "semantic_from_chunks.jsonl"
    align = tmp_path / "paper_repo_align.jsonl"
    spans = tmp_path / "paper_repo_span_align.jsonl"

    _write_jsonl(
        repo_semantic,
        [
            {
                "id": "repoA:repo_chunks",
                "entity_id": "repoA",
                "time_window": "",
                "scope": "repo_chunks",
                "summary_text": "Repo A covers generic infrastructure.",
                "key_concepts": ["infra", "runtime"],
                "dense": [1.0],
            },
            {
                "id": "repoB:repo_chunks",
                "entity_id": "repoB",
                "time_window": "",
                "scope": "repo_chunks",
                "summary_text": "Repo B has generic components.",
                "key_concepts": ["components"],
                "dense": [1.0],
            },
        ],
    )

    _write_jsonl(
        align,
        [
            {
                "paper_id": "pB",
                "paper_title": "Browser Agents with Iterative Exploration",
                "paper_abstract": "",
                "pdf_path": "/arxiv/pdfs/pB.pdf",
                "paper_text": "This work studies browser agents and iterative exploration over websites.",
                "repo_id": "repoB",
                "repo_path": "/data/repositories/repoB/agent.py",
                "repo_offset": 0,
                "repo_text": "browser agent code",
                "label": 1,
                "score": 0.7,
                "shared_terms": ["browser", "agent", "exploration"],
                "candidate_rank": 1,
                "negative_type": None,
            }
        ],
    )

    _write_jsonl(
        spans,
        [
            {
                "paper_id": "pB",
                "paper_title": "Browser Agents with Iterative Exploration",
                "pdf_path": "/arxiv/pdfs/pB.pdf",
                "repo_id": "repoB",
                "repo_path": "/data/repositories/repoB/agent.py",
                "repo_offset": 0,
                "paragraph_id": 1,
                "page_start": 1,
                "page_end": 1,
                "line_start": 1,
                "line_end": 4,
                "paper_text": "Browser agents use iterative exploration and feedback to operate on websites.",
                "repo_text": "browser automation agent",
                "label": 1,
                "score": 0.55,
                "shared_terms": ["browser", "agents", "exploration", "feedback"],
                "negative_type": None,
            }
        ],
    )

    retriever = CoarseLaneRetriever(
        repo_semantic_path=str(repo_semantic),
        paper_repo_align_path=str(align),
        paper_repo_span_path=str(spans),
    )
    result = retriever.retrieve("browser agent iterative exploration", top_k_repos=2, top_k_papers=2, top_k_spans=2)

    assert result["paper_hits"][0]["repo_id"] == "repoB"
    assert result["fused_repos"][0]["repo_id"] == "repoB"
