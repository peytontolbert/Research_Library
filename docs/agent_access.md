# Agent Access

The easiest way to expose the Repository Library to agent clients is the new
stdio MCP server:

```bash
PYTHONPATH=/data/repository_library /home/peyton/miniconda3/envs/ai/bin/python -m scripts.repo_library_mcp
```

The server runs in-process against the current library state under
`/data/repository_library/exports`. It does not require the FastAPI server to
already be running.

The default MCP config generator also prefers the `ai` conda env when it is
available, because that env has the working `torch`/`transformers` runtime for
repo QA on this machine.

## Generic MCP Config

You can print a ready-to-paste generic config block with:

```bash
python -m scripts.repo_library_mcp --print-config-json
```

That emits a standard `mcpServers` block like:

```json
{
  "mcpServers": {
    "repository-library": {
      "command": "/usr/bin/python3",
      "args": ["-m", "scripts.repo_library_mcp"],
      "env": {
        "PYTHONPATH": "/data/repository_library",
        "REPO_LIBRARY_MCP_PYTHON": "/home/peyton/miniconda3/envs/ai/bin/python"
      }
    }
  }
}
```

If your agent client uses a different config format, the important part is the
same:

- command: preferably `/home/peyton/miniconda3/envs/ai/bin/python`
- args: `-m scripts.repo_library_mcp`
- env: `PYTHONPATH=/data/repository_library`

If you need a different interpreter later, set `REPO_LIBRARY_MCP_PYTHON` in
the client config and rerun `--print-config-json`.

This is the path you want for Codex, Claude Code, and any other MCP-capable
client. Clients that do not support MCP can still call the existing HTTP API
from [run.py](/data/repository_library/run.py:2416).

## Recommended Tool Flow

For agents doing repository work:

1. `find_relevant_repos`
2. `get_repo` or `get_repo_profile`
3. `list_repo_skills`
4. `build_repo_skill` if `qa` is missing or stale
5. `repository_qa`
6. `get_repo_graph` and `get_source_snippet` for verification or deeper grounding

For research-heavy flows:

1. `find_relevant_repos`
2. `search_arxiv_papers`
3. `download_arxiv_pdfs` if needed
4. `repository_qa`

For algorithm/library exploration:

1. `search_algorithms`
2. `get_algorithm_problem`
3. `get_algorithm_implementations`

## Exposed MCP Tools

- `list_repos`
- `get_repo`
- `get_repo_profile`
- `list_repo_skills`
- `build_repo_skill`
- `repository_qa`
- `find_relevant_repos`
- `get_repo_graph`
- `get_source_snippet`
- `search_arxiv_papers`
- `download_arxiv_pdfs`
- `search_algorithms`
- `get_algorithm_problem`
- `get_algorithm_implementations`

## Notes

- `repository_qa` supports `auto_build=true`, so an agent can repair a missing
  QA skill inline instead of failing the first query.
- The server changes into the repository root on startup so the existing
  relative paths used by the library keep working.
- The current library still routes QA to an explicit `repo_id`; `find_relevant_repos`
  is the best first step when the agent does not already know the target repo.
