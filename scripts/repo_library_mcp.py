from __future__ import annotations

"""
Minimal stdio MCP server for the Repository Library.

This wraps the existing in-process library/runtime functions so agent clients
such as Codex, Claude Code, and other MCP-capable tools can browse repos,
inspect graphs/source, and run grounded repo QA without standing up a separate
HTTP bridge.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


JSONRPC_VERSION = "2.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "repository-library"
SERVER_VERSION = "0.1.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_AI_PYTHON = Path("/home/peyton/miniconda3/envs/ai/bin/python")


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data


class ToolInvocationError(Exception):
    pass


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _preferred_python_executable() -> str:
    explicit = str(os.environ.get("REPO_LIBRARY_MCP_PYTHON") or "").strip()
    if explicit:
        return explicit
    if PREFERRED_AI_PYTHON.exists():
        return str(PREFERRED_AI_PYTHON)
    return sys.executable


def _schema(
    properties: Dict[str, Any],
    *,
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": False,
    }


def _coerce_int(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    default: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return default
    try:
        num = int(value)
    except Exception as exc:
        raise ToolInvocationError(f"`{field}` must be an integer.") from exc
    if minimum is not None and num < minimum:
        raise ToolInvocationError(f"`{field}` must be >= {minimum}.")
    if maximum is not None and num > maximum:
        raise ToolInvocationError(f"`{field}` must be <= {maximum}.")
    return num


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


_RUN_MODULE = None


def _get_run_module():
    global _RUN_MODULE
    if _RUN_MODULE is None:
        import run as run_module

        _RUN_MODULE = run_module
    return _RUN_MODULE


def _call_async_endpoint(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return asyncio.run(func(*args, **kwargs))
    except Exception:
        run_module = _get_run_module()
        http_exception = getattr(run_module, "HTTPException", None)
        if http_exception is not None:
            exc = sys.exc_info()[1]
            if isinstance(exc, http_exception):
                detail = getattr(exc, "detail", str(exc))
                status_code = getattr(exc, "status_code", "error")
                raise ToolInvocationError(f"{status_code}: {detail}") from None
        raise


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]


class RepositoryLibraryBackend:
    def list_repos(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        run_module = _get_run_module()
        result = _call_async_endpoint(run_module.api_list_repos)
        repos = list(result.get("repos") or [])

        prefix = str(arguments.get("repo_id_prefix") or "").strip()
        if prefix:
            repos = [row for row in repos if str(row.get("repo_id") or "").startswith(prefix)]

        if _coerce_bool(arguments.get("only_with_skills")):
            repos = [row for row in repos if bool(row.get("has_skills"))]
        if _coerce_bool(arguments.get("only_with_extensions")):
            repos = [row for row in repos if bool(row.get("has_extensions"))]
        if _coerce_bool(arguments.get("only_with_repo_skills_miner")):
            repos = [row for row in repos if bool(row.get("has_repo_skills_miner"))]

        limit = _coerce_int(arguments.get("limit"), field="limit", minimum=1, maximum=500)
        if limit is not None:
            repos = repos[:limit]

        return {"count": len(repos), "repos": repos}

    def get_repo(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(arguments.get("repo_id") or "").strip()
        if not repo_id:
            raise ToolInvocationError("`repo_id` is required.")
        run_module = _get_run_module()
        return _call_async_endpoint(run_module.api_get_repo, repo_id)

    def get_repo_profile(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(arguments.get("repo_id") or "").strip()
        if not repo_id:
            raise ToolInvocationError("`repo_id` is required.")
        limit = _coerce_int(arguments.get("limit"), field="limit", minimum=1, maximum=100, default=20)
        run_module = _get_run_module()
        return _call_async_endpoint(
            run_module.api_get_repo_skills_miner_extension,
            repo_id,
            limit=limit,
        )

    def list_repo_skills(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(arguments.get("repo_id") or "").strip()
        if not repo_id:
            raise ToolInvocationError("`repo_id` is required.")
        run_module = _get_run_module()
        return _call_async_endpoint(run_module.api_skills_for_repo, repo_id)

    def build_repo_skill(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(arguments.get("repo_id") or "").strip()
        skill = str(arguments.get("skill") or "").strip()
        if not repo_id or not skill:
            raise ToolInvocationError("`repo_id` and `skill` are required.")
        force = _coerce_bool(arguments.get("force"))
        run_module = _get_run_module()
        return _call_async_endpoint(
            run_module.api_skill_build,
            {"repo_id": repo_id, "skill": skill, "force": force},
        )

    def repository_qa(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(arguments.get("repo_id") or "").strip()
        question = str(arguments.get("question") or "").strip()
        qa_mode_raw = arguments.get("qa_mode")
        qa_mode = str(qa_mode_raw).strip() if isinstance(qa_mode_raw, str) else None
        auto_build = _coerce_bool(arguments.get("auto_build"))

        if not repo_id:
            raise ToolInvocationError("`repo_id` is required.")
        if not question:
            raise ToolInvocationError("`question` is required.")

        if auto_build:
            skill_status = self.list_repo_skills({"repo_id": repo_id})
            qa_rows = [
                row
                for row in (skill_status.get("skills") or [])
                if isinstance(row, dict) and row.get("skill") == "qa"
            ]
            needs_build = not qa_rows or qa_rows[0].get("status") != "up_to_date"
            if needs_build:
                self.build_repo_skill({"repo_id": repo_id, "skill": "qa"})

        run_module = _get_run_module()
        return _call_async_endpoint(
            run_module.api_qa_execute,
            {"question": question, "repo_hint": repo_id, "qa_mode": qa_mode},
        )

    def find_relevant_repos(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        question = str(arguments.get("question") or "").strip()
        if not question:
            raise ToolInvocationError("`question` is required.")
        payload = {
            "question": question,
            "top_k_repos": _coerce_int(
                arguments.get("top_k_repos"),
                field="top_k_repos",
                minimum=1,
                maximum=20,
                default=5,
            ),
            "top_k_papers": _coerce_int(
                arguments.get("top_k_papers"),
                field="top_k_papers",
                minimum=1,
                maximum=20,
                default=5,
            ),
            "top_k_spans": _coerce_int(
                arguments.get("top_k_spans"),
                field="top_k_spans",
                minimum=1,
                maximum=30,
                default=6,
            ),
        }
        run_module = _get_run_module()
        return _call_async_endpoint(run_module.api_coarse_retrieve, payload)

    def get_repo_graph(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(arguments.get("repo_id") or "").strip()
        if not repo_id:
            raise ToolInvocationError("`repo_id` is required.")
        max_nodes = _coerce_int(arguments.get("max_nodes"), field="max_nodes", minimum=1, maximum=5000, default=800)
        max_edges = _coerce_int(arguments.get("max_edges"), field="max_edges", minimum=1, maximum=10000, default=1600)
        run_module = _get_run_module()
        return _call_async_endpoint(
            run_module.api_graph,
            repo_id,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def get_source_snippet(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(arguments.get("repo_id") or "").strip()
        uri = str(arguments.get("uri") or "").strip()
        if not repo_id or not uri:
            raise ToolInvocationError("`repo_id` and `uri` are required.")
        context = _coerce_int(arguments.get("context"), field="context", minimum=0, maximum=200, default=20)
        max_lines = _coerce_int(arguments.get("max_lines"), field="max_lines", minimum=1, maximum=800, default=400)
        run_module = _get_run_module()
        return _call_async_endpoint(
            run_module.api_source,
            repo_id,
            uri=uri,
            context=context,
            max_lines=max_lines,
        )

    def search_arxiv_papers(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolInvocationError("`query` is required.")
        payload: Dict[str, Any] = {"query": query}

        max_results = _coerce_int(arguments.get("max_results"), field="max_results", minimum=1)
        if max_results is not None:
            payload["max_results"] = max_results

        fields = arguments.get("fields")
        if fields is not None:
            if not isinstance(fields, list) or not all(isinstance(x, str) for x in fields):
                raise ToolInvocationError("`fields` must be a list of strings.")
            payload["fields"] = fields

        category_prefix = arguments.get("category_prefix")
        if isinstance(category_prefix, str) and category_prefix.strip():
            payload["category_prefix"] = category_prefix.strip()

        run_module = _get_run_module()
        return _call_async_endpoint(run_module.api_arxiv_search, payload)

    def download_arxiv_pdfs(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ids = arguments.get("ids")
        single_id = arguments.get("id")
        payload: Dict[str, Any] = {}
        if isinstance(single_id, str) and single_id.strip():
            payload["id"] = single_id.strip()
        if ids is not None:
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                raise ToolInvocationError("`ids` must be a list of strings.")
            payload["ids"] = [x.strip() for x in ids if x.strip()]
        if not payload:
            raise ToolInvocationError("`id` or `ids` is required.")
        run_module = _get_run_module()
        return _call_async_endpoint(run_module.api_arxiv_download, payload)

    def search_algorithms(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}

        query = arguments.get("query")
        if isinstance(query, str):
            payload["query"] = query

        problem_id = arguments.get("problem_id")
        if isinstance(problem_id, str) and problem_id.strip():
            payload["problem_id"] = problem_id.strip()

        topic = arguments.get("topic")
        if isinstance(topic, str) and topic.strip():
            payload["topic"] = topic.strip()

        page = _coerce_int(arguments.get("page"), field="page", minimum=1, default=1)
        page_size = _coerce_int(arguments.get("page_size"), field="page_size", minimum=1, maximum=50, default=25)
        payload["page"] = page
        payload["page_size"] = page_size

        run_module = _get_run_module()
        return _call_async_endpoint(run_module.api_algorithms_search, payload)

    def get_algorithm_problem(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        problem_id = str(arguments.get("problem_id") or "").strip()
        if not problem_id:
            raise ToolInvocationError("`problem_id` is required.")
        run_module = _get_run_module()
        return _call_async_endpoint(run_module.api_algorithm_problem, problem_id)

    def get_algorithm_implementations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        algo_id = str(arguments.get("algo_id") or "").strip()
        if not algo_id:
            raise ToolInvocationError("`algo_id` is required.")
        max_results = _coerce_int(arguments.get("max_results"), field="max_results", minimum=1, maximum=200, default=50)
        run_module = _get_run_module()
        return _call_async_endpoint(
            run_module.api_algorithm_implementations,
            algo_id=algo_id,
            max_results=max_results,
        )


class RepositoryLibraryMCPServer:
    def __init__(
        self,
        *,
        backend: Optional[RepositoryLibraryBackend] = None,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        server_name: str = SERVER_NAME,
    ) -> None:
        self._backend = backend or RepositoryLibraryBackend()
        self._protocol_version = protocol_version
        self._server_name = server_name
        self._shutdown_requested = False
        self._tools = self._build_tool_specs()

    def _build_tool_specs(self) -> Dict[str, ToolSpec]:
        return {
            "list_repos": ToolSpec(
                name="list_repos",
                description="List repositories in the library, with optional prefix and capability filters.",
                input_schema=_schema(
                    {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                        "repo_id_prefix": {"type": "string"},
                        "only_with_skills": {"type": "boolean"},
                        "only_with_extensions": {"type": "boolean"},
                        "only_with_repo_skills_miner": {"type": "boolean"},
                    }
                ),
                handler=self._backend.list_repos,
            ),
            "get_repo": ToolSpec(
                name="get_repo",
                description="Fetch manifest metadata for a single repository, including branch, head, and commit count.",
                input_schema=_schema(
                    {"repo_id": {"type": "string", "minLength": 1}},
                    required=["repo_id"],
                ),
                handler=self._backend.get_repo,
            ),
            "get_repo_profile": ToolSpec(
                name="get_repo_profile",
                description="Fetch the structured repo_skills_miner profile for a repository, including summary, skills, annotations, and signals.",
                input_schema=_schema(
                    {
                        "repo_id": {"type": "string", "minLength": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    required=["repo_id"],
                ),
                handler=self._backend.get_repo_profile,
            ),
            "list_repo_skills": ToolSpec(
                name="list_repo_skills",
                description="List build status for the known per-repository skills.",
                input_schema=_schema(
                    {"repo_id": {"type": "string", "minLength": 1}},
                    required=["repo_id"],
                ),
                handler=self._backend.list_repo_skills,
            ),
            "build_repo_skill": ToolSpec(
                name="build_repo_skill",
                description="Build or refresh a repository skill such as qa.",
                input_schema=_schema(
                    {
                        "repo_id": {"type": "string", "minLength": 1},
                        "skill": {"type": "string", "minLength": 1},
                        "force": {"type": "boolean"},
                    },
                    required=["repo_id", "skill"],
                ),
                handler=self._backend.build_repo_skill,
            ),
            "repository_qa": ToolSpec(
                name="repository_qa",
                description="Run grounded QA for a repository. Optionally auto-build the qa skill first if it is missing or stale.",
                input_schema=_schema(
                    {
                        "repo_id": {"type": "string", "minLength": 1},
                        "question": {"type": "string", "minLength": 1},
                        "qa_mode": {"type": "string"},
                        "auto_build": {"type": "boolean"},
                    },
                    required=["repo_id", "question"],
                ),
                handler=self._backend.repository_qa,
            ),
            "find_relevant_repos": ToolSpec(
                name="find_relevant_repos",
                description="Use paper-repo coarse retrieval to suggest relevant repositories for a question.",
                input_schema=_schema(
                    {
                        "question": {"type": "string", "minLength": 1},
                        "top_k_repos": {"type": "integer", "minimum": 1, "maximum": 20},
                        "top_k_papers": {"type": "integer", "minimum": 1, "maximum": 20},
                        "top_k_spans": {"type": "integer", "minimum": 1, "maximum": 30},
                    },
                    required=["question"],
                ),
                handler=self._backend.find_relevant_repos,
            ),
            "get_repo_graph": ToolSpec(
                name="get_repo_graph",
                description="Fetch a sampled node and edge view of a repository graph for navigation or visualization.",
                input_schema=_schema(
                    {
                        "repo_id": {"type": "string", "minLength": 1},
                        "max_nodes": {"type": "integer", "minimum": 1, "maximum": 5000},
                        "max_edges": {"type": "integer", "minimum": 1, "maximum": 10000},
                    },
                    required=["repo_id"],
                ),
                handler=self._backend.get_repo_graph,
            ),
            "get_source_snippet": ToolSpec(
                name="get_source_snippet",
                description="Resolve a program URI to the underlying file snippet and line span.",
                input_schema=_schema(
                    {
                        "repo_id": {"type": "string", "minLength": 1},
                        "uri": {"type": "string", "minLength": 1},
                        "context": {"type": "integer", "minimum": 0, "maximum": 200},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 800},
                    },
                    required=["repo_id", "uri"],
                ),
                handler=self._backend.get_source_snippet,
            ),
            "search_arxiv_papers": ToolSpec(
                name="search_arxiv_papers",
                description="Search the local arXiv metadata snapshot and report whether local PDFs are already present.",
                input_schema=_schema(
                    {
                        "query": {"type": "string", "minLength": 1},
                        "max_results": {"type": "integer", "minimum": 1},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "category_prefix": {"type": "string"},
                    },
                    required=["query"],
                ),
                handler=self._backend.search_arxiv_papers,
            ),
            "download_arxiv_pdfs": ToolSpec(
                name="download_arxiv_pdfs",
                description="Download one or more arXiv PDFs into the local cache.",
                input_schema=_schema(
                    {
                        "id": {"type": "string"},
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    }
                ),
                handler=self._backend.download_arxiv_pdfs,
            ),
            "search_algorithms": ToolSpec(
                name="search_algorithms",
                description="Search the local algorithms library by free-text query, problem id, or topic.",
                input_schema=_schema(
                    {
                        "query": {"type": "string"},
                        "problem_id": {"type": "string"},
                        "topic": {"type": "string"},
                        "page": {"type": "integer", "minimum": 1},
                        "page_size": {"type": "integer", "minimum": 1, "maximum": 50},
                    }
                ),
                handler=self._backend.search_algorithms,
            ),
            "get_algorithm_problem": ToolSpec(
                name="get_algorithm_problem",
                description="Fetch a single algorithms-library problem by problem id.",
                input_schema=_schema(
                    {"problem_id": {"type": "string", "minLength": 1}},
                    required=["problem_id"],
                ),
                handler=self._backend.get_algorithm_problem,
            ),
            "get_algorithm_implementations": ToolSpec(
                name="get_algorithm_implementations",
                description="List concrete implementations for an algorithm across the library.",
                input_schema=_schema(
                    {
                        "algo_id": {"type": "string", "minLength": 1},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    required=["algo_id"],
                ),
                handler=self._backend.get_algorithm_implementations,
            ),
        }

    def _make_response(self, request_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    def _make_error_response(self, request_id: Any, error: MCPError) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": error.code, "message": error.message},
        }
        if error.data is not None:
            payload["error"]["data"] = error.data
        return payload

    def _initialize_result(self) -> Dict[str, Any]:
        return {
            "protocolVersion": self._protocol_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": self._server_name, "version": SERVER_VERSION},
        }

    def _tools_list_result(self) -> Dict[str, Any]:
        tools = []
        for spec in self._tools.values():
            tools.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.input_schema,
                }
            )
        return {"tools": tools}

    def _tool_call_result(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            raise MCPError(METHOD_NOT_FOUND, f"Unknown tool: {name}")
        try:
            value = spec.handler(arguments)
            structured: Any
            if isinstance(value, dict):
                structured = value
            else:
                structured = {"result": value}
            return {
                "content": [{"type": "text", "text": _json_text(structured)}],
                "structuredContent": structured,
                "isError": False,
            }
        except ToolInvocationError as exc:
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }

    def handle_request(self, method: str, params: Optional[Dict[str, Any]]) -> Any:
        params = params or {}
        if method == "initialize":
            return self._initialize_result()
        if method == "ping":
            return {}
        if method == "tools/list":
            return self._tools_list_result()
        if method == "tools/call":
            name = str(params.get("name") or "").strip()
            arguments = params.get("arguments") or {}
            if not name:
                raise MCPError(INVALID_PARAMS, "`name` is required for tools/call.")
            if not isinstance(arguments, dict):
                raise MCPError(INVALID_PARAMS, "`arguments` must be an object.")
            return self._tool_call_result(name, arguments)
        if method == "resources/list":
            return {"resources": []}
        if method == "prompts/list":
            return {"prompts": []}
        if method == "shutdown":
            self._shutdown_requested = True
            return {}
        raise MCPError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def handle_notification(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        _ = params
        if method == "exit":
            self._shutdown_requested = True
        # Ignore `notifications/initialized` and other non-request messages.

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(message, dict):
            raise MCPError(INVALID_REQUEST, "Request payload must be an object.")

        method = message.get("method")
        if not isinstance(method, str) or not method:
            raise MCPError(INVALID_REQUEST, "Request is missing a valid `method`.")

        request_id = message.get("id")
        params = message.get("params")
        if params is not None and not isinstance(params, dict):
            raise MCPError(INVALID_PARAMS, "`params` must be an object.")

        if request_id is None:
            self.handle_notification(method, params)
            return None

        result = self.handle_request(method, params)
        return self._make_response(request_id, result)

    def should_exit(self) -> bool:
        return self._shutdown_requested


def _read_message(stdin: Any) -> Optional[Dict[str, Any]]:
    headers: Dict[str, str] = {}
    while True:
        line = stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MCPError(PARSE_ERROR, f"Failed to decode header line: {exc}") from exc
        if ":" not in text:
            raise MCPError(INVALID_REQUEST, f"Malformed header line: {text.strip()!r}")
        key, value = text.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    length_text = headers.get("content-length")
    if not length_text:
        raise MCPError(INVALID_REQUEST, "Missing Content-Length header.")
    try:
        length = int(length_text)
    except ValueError as exc:
        raise MCPError(INVALID_REQUEST, f"Invalid Content-Length: {length_text!r}") from exc

    body = stdin.buffer.read(length)
    if len(body) != length:
        raise MCPError(PARSE_ERROR, "Truncated JSON-RPC body.")
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MCPError(PARSE_ERROR, f"Failed to decode request body: {exc}") from exc
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise MCPError(PARSE_ERROR, f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise MCPError(INVALID_REQUEST, "Top-level JSON-RPC payload must be an object.")
    return payload


def _write_message(stdout: Any, message: Dict[str, Any]) -> None:
    encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
    stdout.buffer.write(header)
    stdout.buffer.write(encoded)
    stdout.buffer.flush()


def build_generic_mcp_config(*, server_name: str = SERVER_NAME) -> Dict[str, Any]:
    return {
        "mcpServers": {
            server_name: {
                "command": _preferred_python_executable(),
                "args": ["-m", "scripts.repo_library_mcp"],
                "env": {
                    "PYTHONPATH": str(REPO_ROOT),
                    "REPO_LIBRARY_MCP_PYTHON": _preferred_python_executable(),
                },
            }
        }
    }


def run_stdio_server(*, server_name: str = SERVER_NAME) -> int:
    os.chdir(REPO_ROOT)
    server = RepositoryLibraryMCPServer(server_name=server_name)

    while not server.should_exit():
        try:
            message = _read_message(sys.stdin)
            if message is None:
                break
            response = server.process_message(message)
            if response is not None:
                _write_message(sys.stdout, response)
        except MCPError as exc:
            request_id = None
            _write_message(
                sys.stdout,
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": request_id,
                    "error": {"code": exc.code, "message": exc.message},
                },
            )
        except KeyboardInterrupt:
            break
        except Exception as exc:
            _write_message(
                sys.stdout,
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": None,
                    "error": {
                        "code": INTERNAL_ERROR,
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                },
            )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Repository Library MCP server over stdio.")
    parser.add_argument(
        "--server-name",
        default=SERVER_NAME,
        help="Server name to advertise in MCP initialize responses.",
    )
    parser.add_argument(
        "--print-config-json",
        action="store_true",
        help="Print a generic stdio MCP config block for this server and exit.",
    )
    args = parser.parse_args(argv)

    if args.print_config_json:
        print(_json_text(build_generic_mcp_config(server_name=args.server_name)))
        return 0

    return run_stdio_server(server_name=args.server_name)


if __name__ == "__main__":
    raise SystemExit(main())
