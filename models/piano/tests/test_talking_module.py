from models.piano.modules import TalkingModule


class DummyLLM:
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompts, max_new_tokens=0, temperature=0.0):
        return [self.text for _ in prompts]


def test_talking_module_truncates_to_three_bullets():
    long_reply = "- one\n- two\n- three\n- four\n"
    tm = TalkingModule(llm=DummyLLM(long_reply))
    out = tm.respond({"semantic_context": "", "episodic_context": "", "user_task": "do thing", "system": ""})
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 3
    assert lines == ["- one", "- two", "- three"]
