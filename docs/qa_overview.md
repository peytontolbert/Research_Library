## QA Skill Overview

The QA skill provides **model-based question answering over code** for a
single repository. It combines:

- The exported **program graph** (entities, edges, artifacts).
- A per-repo **vector index** of entities for similarity search.
- A per-repo **QA adapter** that specifies which model to use.
- A shared **QA runtime** that loads models (optionally 4‑bit) and runs
  generation.

At a high level:

1. Exports and `build_skill(repo_id, "qa")` prepare graph + index +
   adapter metadata.
2. At runtime, `/api/skill_chat` or `/api/qa_execute`:
   - Plan a QA query via `RepoLibrary.query`.
   - Use the QA adapter to drive retrieval + model selection.
   - Call the QA runtime to generate a chat-quality answer.


