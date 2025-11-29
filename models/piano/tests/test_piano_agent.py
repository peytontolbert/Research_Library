import tempfile
from pathlib import Path

from models.piano.agent import PianoAgent
from models.piano.modules import SkillExecutor


def test_skill_executor_edit_and_diff(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True, exist_ok=True)
    execu = SkillExecutor(base_repo_root=tmp_path)
    # should create file and return diff snippet
    res = execu.edit_file("repo", "foo.txt", "hello")
    assert "edit_file:repo/foo.txt" in res
    assert (repo_dir / "foo.txt").exists()
    assert "diff" in res or "no-diff" in res


def test_piano_agent_step_runs(tmp_path: Path, monkeypatch):
    # point SkillExecutor to temp repo and avoid git diff call errors
    agent = PianoAgent(task="run tests", repo_id=None, paper_id=None)
    # override skill executor to avoid touching real filesystem
    agent.skills.base_repo_root = tmp_path
    out = agent.step()
    assert "intent" in out
    assert "action_result" in out


def test_skill_executor_ci_and_graph(tmp_path: Path):
    repo_dir = tmp_path / "repo2"
    (repo_dir / ".git").mkdir(parents=True, exist_ok=True)
    def fake_runner(path: Path, cmd):
        return f"[ci:{path.name}]"
    class FakeGraph:
        def expand(self, concept: str):
            return [{"concept_id": concept + "_n1"}, {"concept_id": concept + "_n2"}]
    execu = SkillExecutor(base_repo_root=tmp_path, ci_runner=fake_runner, graph_client=fake_graph)
    ci_res = execu.run_ci("repo2", ["echo", "ok"])
    graph_res = execu.graph_neighbors("concept_x")
    assert "ci:repo2" in ci_res
    assert "concept_x_n1" in graph_res

def test_skill_executor_default_ci(tmp_path: Path):
    repo_dir = tmp_path / "repo3"
    (repo_dir / ".git").mkdir(parents=True, exist_ok=True)
    (repo_dir / "scripts").mkdir(parents=True, exist_ok=True)
    ci_script = repo_dir / "scripts" / "ci.sh"
    ci_script.write_text("echo ci-ok\n", encoding="utf-8")
    execu = SkillExecutor(base_repo_root=tmp_path)
    res = execu.run_ci("repo3")
    assert "run_ci:repo3" in res


def test_apply_lora_and_fine_tune_missing(tmp_path: Path):
    execu = SkillExecutor(base_repo_root=tmp_path)
    assert "missing" in execu.apply_lora("/tmp/does_not_exist")
    assert "missing" in execu.fine_tune("repo", "/tmp/does_not_exist")


def test_env_override_commands(monkeypatch, tmp_path: Path):
    cmd = '["echo","ok"]'
    monkeypatch.setenv("PIANO_FINE_TUNE_CMD", cmd)
    execu = SkillExecutor(base_repo_root=tmp_path)
    res = execu.fine_tune("repo", str(tmp_path / "cfg.json"))
    assert "fine_tune_missing" in res  # config missing
    monkeypatch.setenv("PIANO_APPLY_LORA_CMD", cmd)
    res = execu.apply_lora(str(tmp_path / "adapter.py"))
    assert "apply_lora_missing" in res  # adapter missing
