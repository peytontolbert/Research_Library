from __future__ import annotations

import json

from scripts.repo_library_mcp import (
    DEFAULT_PROTOCOL_VERSION,
    RepositoryLibraryMCPServer,
    ToolInvocationError,
    build_generic_mcp_config,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return {"tool": name, "arguments": dict(arguments)}

    def list_repos(self, arguments: dict) -> dict:
        return self._record("list_repos", arguments)

    def get_repo(self, arguments: dict) -> dict:
        return self._record("get_repo", arguments)

    def get_repo_profile(self, arguments: dict) -> dict:
        return self._record("get_repo_profile", arguments)

    def list_repo_skills(self, arguments: dict) -> dict:
        return self._record("list_repo_skills", arguments)

    def build_repo_skill(self, arguments: dict) -> dict:
        return self._record("build_repo_skill", arguments)

    def repository_qa(self, arguments: dict) -> dict:
        if arguments.get("question") == "fail":
            raise ToolInvocationError("qa failed")
        return self._record("repository_qa", arguments)

    def find_relevant_repos(self, arguments: dict) -> dict:
        return self._record("find_relevant_repos", arguments)

    def get_repo_graph(self, arguments: dict) -> dict:
        return self._record("get_repo_graph", arguments)

    def get_source_snippet(self, arguments: dict) -> dict:
        return self._record("get_source_snippet", arguments)

    def search_arxiv_papers(self, arguments: dict) -> dict:
        return self._record("search_arxiv_papers", arguments)

    def download_arxiv_pdfs(self, arguments: dict) -> dict:
        return self._record("download_arxiv_pdfs", arguments)

    def search_algorithms(self, arguments: dict) -> dict:
        return self._record("search_algorithms", arguments)

    def get_algorithm_problem(self, arguments: dict) -> dict:
        return self._record("get_algorithm_problem", arguments)

    def get_algorithm_implementations(self, arguments: dict) -> dict:
        return self._record("get_algorithm_implementations", arguments)


def test_initialize_and_tools_list() -> None:
    server = RepositoryLibraryMCPServer(backend=_FakeBackend())

    init = server.process_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": DEFAULT_PROTOCOL_VERSION},
        }
    )
    assert init is not None
    assert init["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
    assert init["result"]["capabilities"]["tools"]["listChanged"] is False

    tool_list = server.process_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    assert tool_list is not None
    names = {tool["name"] for tool in tool_list["result"]["tools"]}
    assert "list_repos" in names
    assert "repository_qa" in names
    assert "get_source_snippet" in names
    assert "search_algorithms" in names


def test_tools_call_routes_to_backend_and_returns_structured_content() -> None:
    backend = _FakeBackend()
    server = RepositoryLibraryMCPServer(backend=backend)

    response = server.process_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_repo",
                "arguments": {"repo_id": "demo"},
            },
        }
    )

    assert response is not None
    result = response["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["tool"] == "get_repo"
    assert result["structuredContent"]["arguments"] == {"repo_id": "demo"}
    assert backend.calls == [("get_repo", {"repo_id": "demo"})]

    parsed = json.loads(result["content"][0]["text"])
    assert parsed["tool"] == "get_repo"


def test_tools_call_returns_tool_error_without_protocol_failure() -> None:
    server = RepositoryLibraryMCPServer(backend=_FakeBackend())

    response = server.process_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "repository_qa",
                "arguments": {"repo_id": "demo", "question": "fail"},
            },
        }
    )

    assert response is not None
    result = response["result"]
    assert result["isError"] is True
    assert "qa failed" in result["content"][0]["text"]


def test_exit_notification_sets_shutdown_flag() -> None:
    server = RepositoryLibraryMCPServer(backend=_FakeBackend())
    assert server.should_exit() is False
    assert server.process_message({"jsonrpc": "2.0", "method": "exit"}) is None
    assert server.should_exit() is True


def test_build_generic_mcp_config_uses_module_entrypoint(monkeypatch) -> None:
    monkeypatch.setenv("REPO_LIBRARY_MCP_PYTHON", "/tmp/custom-python")
    cfg = build_generic_mcp_config(server_name="repo-library")
    server_cfg = cfg["mcpServers"]["repo-library"]
    assert server_cfg["command"] == "/tmp/custom-python"
    assert server_cfg["args"] == ["-m", "scripts.repo_library_mcp"]
    assert "PYTHONPATH" in server_cfg["env"]
    assert server_cfg["env"]["REPO_LIBRARY_MCP_PYTHON"] == "/tmp/custom-python"
