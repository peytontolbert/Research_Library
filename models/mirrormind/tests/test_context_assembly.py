from models.mirrormind.context import ContextAssembler
from models.mirrormind.memory import Episode, SemanticSummary


class DummyTwin:
    def __init__(self) -> None:
        self._semantic = [
            SemanticSummary(
                id="s1",
                entity_id="e1",
                time_window="",
                scope="subsystem",
                summary_text="alpha beta gamma",
                key_concepts=[],
                dense=[],
            ),
            # duplicate summary to test deduplication
            SemanticSummary(
                id="s2",
                entity_id="e1",
                time_window="",
                scope="subsystem",
                summary_text="alpha beta gamma",
                key_concepts=[],
                dense=[],
            ),
        ]
        long_text = "x" * 400
        self._episodic = [
            Episode(
                id="e1",
                entity_id="e1",
                time=None,
                type="doc_paragraph",
                text=long_text,
                graph_context=[],
                dense=[],
                sparse={},
            )
        ]

    def persona_prompt(self) -> str:
        return "PERSONA"

    def semantic_scope(self, task_text: str, top_k: int = 3):
        return self._semantic[:top_k]

    def episodic_context(self, task_text: str, types=None, top_k: int = 5):
        return self._episodic[:top_k]


def test_context_assembler_dedup_and_truncation():
    twin = DummyTwin()
    assembler = ContextAssembler()

    ctx = assembler.build_repo_context(twin, "do something", max_semantic=5, max_episodic=5)

    assert set(ctx.keys()) == {"system", "semantic_context", "episodic_context", "user_task"}
    assert ctx["system"] == "PERSONA"

    sem_lines = [ln for ln in ctx["semantic_context"].splitlines() if ln.strip()]
    # Duplicate semantic summaries should be collapsed into one line.
    assert len(sem_lines) == 1

    epi_lines = [ln for ln in ctx["episodic_context"].splitlines() if ln.strip()]
    assert len(epi_lines) == 1
    # Episodic text should be truncated and marked with ellipsis.
    assert epi_lines[0].endswith("...")


