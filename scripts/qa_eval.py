from __future__ import annotations

"""
Offline QA evaluation for Repository Library skills.

This module is intentionally *standalone* and can be run as:

    python -m scripts.qa_eval --dataset /path/to/qa.jsonl --condition skill base_rag

It is designed around the metric stack described in the project notes:

- User-level: answer correctness (exact-match style), per-repo accuracy.
- Adapter-level: delta vs a baseline that uses the same retrieval but
  *no* repo/task LoRA adapters (\"base+RAG\").

The evaluator does **not** change the live FastAPI runtime; instead it
reuses the existing RepoLibrary + QA swarm + QA runtime helpers by
importing from `run.py`.

Dataset format
--------------

JSONL file, one object per line:

    {
      "question_id": "uuid-or-name",
      "repo_id": "my_repo",
      "question": "Natural language question...",
      "gold_answer": "Canonical answer text...",
      "gold_evidence": [
        {
          "path": "src/module/file.py",
          "start_line": 10,        # optional
          "end_line": 40           # optional
        }
      ]
    }

Only `question_id`, `repo_id`, `question`, and `gold_answer` are
required. `gold_evidence` is accepted for future retrieval metrics but
is currently unused beyond being logged through.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# We deliberately import from `run` to reuse the exact same QA runtime
# that powers the FastAPI server, including:
# - RepoLibrary instance (`repo_lib`)
# - QA swarm orchestrator (`_qa_swarm_controller`)
# - QA answer stub that wires retrieval + LLM (`_format_qa_answer_stub`)
# - Base LLM helper (`_llm_generate_answer`) for no-adapter baseline.
try:  # pragma: no cover - import side effects are exercised via CLI
    from run import (  # type: ignore
        repo_lib,
        _qa_swarm_controller,
        _format_qa_answer_stub,
        _llm_generate_answer,
    )
    from scripts.repo_library import QueryMode  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    raise RuntimeError(
        "Failed to import QA runtime pieces from `run.py`. "
        "Ensure you are running this module from the project root."
    ) from exc


# -------------------------
# Data structures
# -------------------------


@dataclass
class GoldEvidence:
    path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class QAExample:
    question_id: str
    repo_id: str
    question: str
    gold_answer: str
    gold_evidence: List[GoldEvidence]


@dataclass
class QARunResult:
    example: QAExample
    condition: str
    answer_text: str
    plan: Optional[Dict[str, Any]]
    # Simple user-level metric hooks
    exact_match: float
    # Normalized strings are helpful when debugging failures.
    answer_norm: str
    gold_norm: str


# -------------------------
# Helpers
# -------------------------


def _normalize_text(text: str) -> str:
    """
    Cheap normalization for exact / relaxed match:
    - lowercased
    - stripped
    - collapse internal whitespace
    """
    s = " ".join(str(text or "").split())
    return s.lower()


def _load_dataset(path: Path) -> List[QAExample]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    examples: List[QAExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON on line {lineno}: {exc}") from exc

            qid = str(obj.get("question_id") or "").strip()
            repo_id = str(obj.get("repo_id") or "").strip()
            question = str(obj.get("question") or "").strip()
            gold_answer = str(obj.get("gold_answer") or "").strip()
            if not (qid and repo_id and question and gold_answer):
                raise ValueError(
                    f"missing required fields on line {lineno} "
                    "(need question_id, repo_id, question, gold_answer)"
                )

            evid_raw = obj.get("gold_evidence") or []
            evidence: List[GoldEvidence] = []
            if isinstance(evid_raw, Iterable):
                for e in evid_raw:
                    if not isinstance(e, dict):
                        continue
                    p = str(e.get("path") or "").strip()
                    if not p:
                        continue
                    start = e.get("start_line")
                    end = e.get("end_line")
                    evidence.append(
                        GoldEvidence(
                            path=p,
                            start_line=int(start) if isinstance(start, int) else None,
                            end_line=int(end) if isinstance(end, int) else None,
                        )
                    )

            examples.append(
                QAExample(
                    question_id=qid,
                    repo_id=repo_id,
                    question=question,
                    gold_answer=gold_answer,
                    gold_evidence=evidence,
                )
            )

    return examples


# -------------------------
# Core evaluator
# -------------------------


class QAEvaluator:
    """
    Evaluate the repo-QA skill under different runtime *conditions*.

    Supported conditions:

    - \"skill\":
        Full QA skill via the QASwarmController, i.e. Repo+Task LoRA
        adapters (when present) + retrieval, using the same entrypoint
        as `/api/skill_chat` with `skill="qa"`.

    - \"base_rag\":
        Same retrieval and prompting as the QA skill, but force the
        runtime to use the *base* model (no LoRA adapters) by calling
        `_format_qa_answer_stub(plan, qa_meta=None)`.

    - \"base\":
        No retrieval at all; just run the shared base LLM against the
        raw question. This isolates the value of retrieval itself.
    """

    def __init__(self, examples: List[QAExample]) -> None:
        self.examples = examples

    # --- low-level runners for each condition --- #

    def _run_condition_skill(
        self, ex: QAExample
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Route through the same QA swarm used by the FastAPI server.
        """
        result = _qa_swarm_controller.run_qa(
            question=ex.question,
            repo_hint=ex.repo_id,
            qa_mode=None,
        )
        answer = str(result.get("answer") or "").strip()
        plan = result.get("plan")
        plan_obj = plan if isinstance(plan, dict) else None
        return answer, plan_obj

    def _run_condition_base_rag(
        self, ex: QAExample
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Use RepoLibrary planning + the same retrieval logic as the QA
        skill, but *without* LoRA adapters (base model only).
        """
        plan = repo_lib.query(
            question=ex.question,
            mode=QueryMode.QA,
            repo_hint=ex.repo_id,
            qa_mode=None,
        )
        # Passing qa_meta=None forces `_format_qa_answer_stub` to fall
        # back to the shared base LLM (`_llm_generate_answer`), while
        # still using the same retrieval + prompting pipeline.
        answer = _format_qa_answer_stub(plan, qa_meta=None)
        return str(answer or "").strip(), plan

    def _run_condition_base(
        self, ex: QAExample
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        No retrieval; run the base LLM directly on the question. This
        baseline is intentionally simple and not meant for production.
        """
        prompt = (
            "You are a general-purpose assistant (no code context).\n"
            "Answer the following question as best as you can:\n\n"
            f"{ex.question}\n\n"
            "If the question seems to depend on repository-specific code "
            "that you do not see, answer conservatively and make that "
            "limitation explicit."
        )
        answer = _llm_generate_answer(prompt)
        return str(answer or "").strip(), None

    def _run_single(
        self,
        ex: QAExample,
        condition: str,
    ) -> QARunResult:
        condition_norm = condition.strip().lower()
        if condition_norm == "skill":
            answer, plan = self._run_condition_skill(ex)
        elif condition_norm in ("base_rag", "base+rag", "base-rag"):
            answer, plan = self._run_condition_base_rag(ex)
            condition_norm = "base_rag"
        elif condition_norm == "base":
            answer, plan = self._run_condition_base(ex)
        else:
            raise ValueError(f"unsupported condition: {condition!r}")

        gold_norm = _normalize_text(ex.gold_answer)
        answer_norm = _normalize_text(answer)
        exact_match = 1.0 if (gold_norm and answer_norm == gold_norm) else 0.0

        return QARunResult(
            example=ex,
            condition=condition_norm,
            answer_text=answer,
            plan=plan,
            exact_match=exact_match,
            answer_norm=answer_norm,
            gold_norm=gold_norm,
        )

    # --- public API --- #

    def evaluate(
        self,
        *,
        conditions: List[str],
        max_examples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation for one or more conditions and return aggregated
        metrics + per-example results.

        Returns:
            {
              "conditions": {
                "skill": {
                  "num_examples": N,
                  "accuracy": 0.0-1.0,
                },
                "base_rag": {...},
                ...
              },
              "per_example": [
                {
                  "question_id": ...,
                  "repo_id": ...,
                  "condition": ...,
                  "exact_match": ...,
                  "answer": ...,
                  "gold_answer": ...,
                },
                ...
              ]
            }
        """
        if not conditions:
            raise ValueError("at least one condition is required")

        # Ensure deterministic order.
        conds: List[str] = []
        seen: set[str] = set()
        for c in conditions:
            c_norm = c.strip().lower()
            if not c_norm or c_norm in seen:
                continue
            seen.add(c_norm)
            conds.append(c_norm)

        results: List[QARunResult] = []
        for ex_idx, ex in enumerate(self.examples):
            if max_examples is not None and ex_idx >= max_examples:
                break
            for cond in conds:
                run_res = self._run_single(ex, cond)
                results.append(run_res)

        # Aggregate simple user-level metrics.
        by_cond: Dict[str, Dict[str, Any]] = {}
        for r in results:
            info = by_cond.setdefault(
                r.condition,
                {"num_examples": 0, "sum_exact": 0.0},
            )
            info["num_examples"] += 1
            info["sum_exact"] += float(r.exact_match)

        for cond, info in by_cond.items():
            n = max(1, int(info["num_examples"]))
            info["accuracy"] = info["sum_exact"] / float(n)

        per_example_payload: List[Dict[str, Any]] = []
        for r in results:
            ex = r.example
            per_example_payload.append(
                {
                    "question_id": ex.question_id,
                    "repo_id": ex.repo_id,
                    "condition": r.condition,
                    "exact_match": r.exact_match,
                    "answer": r.answer_text,
                    "gold_answer": ex.gold_answer,
                }
            )

        return {
            "conditions": by_cond,
            "per_example": per_example_payload,
        }


# -------------------------
# CLI
# -------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline evaluator for repo QA skills.")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to QA dataset JSONL file.",
    )
    p.add_argument(
        "--condition",
        type=str,
        nargs="+",
        default=["skill"],
        help=(
            "One or more conditions to evaluate: "
            "skill, base_rag, base (default: skill)."
        ),
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples to run.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write full JSON results; if omitted, prints a summary.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    dataset_path = Path(args.dataset)
    examples = _load_dataset(dataset_path)

    evaluator = QAEvaluator(examples)
    metrics = evaluator.evaluate(
        conditions=list(args.condition or ["skill"]),
        max_examples=args.max_examples,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)
        print(f"Wrote detailed results to {out_path}")
    else:
        # Human-readable summary to stdout.
        print("=== QA Evaluation Summary ===")
        for cond, info in sorted(metrics["conditions"].items()):
            acc = float(info.get("accuracy") or 0.0)
            n = int(info.get("num_examples") or 0)
            print(f"- condition={cond}: accuracy={acc:.3f} over {n} examples")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])


