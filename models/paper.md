1. Abstract

We describe a MirrorMind-style architecture instantiated over your existing repository graph and research paper graph. The system treats each repository or paper as a cognitive entity with three explicit memory systems:

Episodic memory over fine-grained code and text artifacts.

Semantic memory over time-evolving summaries and trajectories.

Persona schema as a concept graph and style descriptor.

These individual memories sit on top of a shared concept/domain graph constructed from code, documentation, and papers. An interdisciplinary orchestration layer routes tasks across project-level and paper-level twins to synthesize plans, explanations, and code.

We specify, for each level, the variables required from each repository/paper and the data structures necessary to implement the architecture on your graph substrate.

2. Knowledge Substrate

We assume you have already constructed:

A ProgramGraph for each repository (functions, files, modules, dependencies).

A RepoGraph (repositories as nodes, similarity edges).

A PaperGraph (papers, citations, topical links).

A ConceptGraph (concepts unified across code and papers, explicit or implicit).

We formalize this universe as:

Set of repositories: 
𝑅
=
{
𝑟
1
,
…
,
𝑟
𝑁
𝑅
}
R={r
1
	​

,…,r
N
R
	​

	​

}

Set of papers: 
𝑃
=
{
𝑝
1
,
…
,
𝑝
𝑁
𝑃
}
P={p
1
	​

,…,p
N
P
	​

	​

}

Set of concepts: 
𝐶
=
{
𝑐
1
,
…
,
𝑐
𝑁
𝐶
}
C={c
1
	​

,…,c
N
C
	​

	​

}

Each repository 
𝑟
∈
𝑅
r∈R and each paper 
𝑝
∈
𝑃
p∈P is mapped into:

Graph neighborhood in the RepoGraph / PaperGraph / ConceptGraph.

Textual artifacts (code, README, docs, abstract, sections).

Temporal evolution (commits, versions; publication dates, revision dates).

2.1 Repository Variables

For each repository 
𝑟
r we maintain:

Identifiers

id
𝑟
id
r
	​

 – unique ID

name
𝑟
name
r
	​

, 
url
𝑟
url
r
	​


Static metadata

languages
𝑟
⊆
{
Python, C++, CUDA, ...
}
languages
r
	​

⊆{Python, C++, CUDA, ...}

tags
𝑟
⊆
𝐶
tags
r
	​

⊆C (high-level concepts: “MoE”, “RL”, “Diffusion”)

stars
𝑟
,
forks
𝑟
,
size
𝑟
stars
r
	​

,forks
r
	​

,size
r
	​

 (optional but useful signals)

Program graph

Nodes 
𝑉
𝑟
=
{
𝑣
1
,
…
,
𝑣
∣
𝑉
𝑟
∣
}
V
r
	​

={v
1
	​

,…,v
∣V
r
	​

∣
	​

} (functions, classes, files, modules)

Edges 
𝐸
𝑟
⊆
𝑉
𝑟
×
𝑉
𝑟
E
r
	​

⊆V
r
	​

×V
r
	​

 (calls, imports, inherits, defines, tests)

Temporal data

Commit sequence 
𝐾
𝑟
=
{
𝑘
1
,
…
,
𝑘
𝑇
𝑟
}
K
r
	​

={k
1
	​

,…,k
T
r
	​

	​

}

Each commit 
𝑘
𝑡
k
t
	​

 has:

timestamp 
𝑡
(
𝑘
𝑡
)
t(k
t
	​

)

diff summary

changed nodes 
Δ
𝑉
𝑟
,
𝑡
⊆
𝑉
𝑟
ΔV
r,t
	​

⊆V
r
	​


Textual artifacts

README, docs, design notes, comments, issue bodies, PR discussions.

Cross-graph links

Linked papers: 
linked_papers
𝑟
⊆
𝑃
linked_papers
r
	​

⊆P

Domain concepts: 
concepts
𝑟
⊆
𝐶
concepts
r
	​

⊆C derived from code and docs.

2.2 Paper Variables

For each paper 
𝑝
p we maintain:

Identifiers

id
𝑝
id
p
	​

, 
doi
𝑝
doi
p
	​

 (if available)

title
𝑝
title
p
	​

, 
venue
𝑝
venue
p
	​


Metadata

Publication year 
year
𝑝
year
p
	​


Authors 
authors
𝑝
=
{
𝑎
1
,
…
,
𝑎
𝑚
}
authors
p
	​

={a
1
	​

,…,a
m
	​

}

Fields / categories 
fields
𝑝
⊆
𝐶
fields
p
	​

⊆C

Textual content

Abstract, sections, equations, pseudo-code blocks.

Graph structure

Citations 
cites
𝑝
⊆
𝑃
cites
p
	​

⊆P

Cited-by edges implicitly from other papers.

Cross-graph links

Referenced repositories 
linked_repos
𝑝
⊆
𝑅
linked_repos
p
	​

⊆R

Concepts 
concepts
𝑝
⊆
𝐶
concepts
p
	​

⊆C from title/abstract/body.

These variables are already compatible with your current graph construction; we now organize them into the MirrorMind-style levels.

3. Individual Level: Project and Paper Twins

At the Individual Level, the system constructs digital twins of repositories and papers:

RepoTwin(r) for each repository 
𝑟
r

PaperTwin(p) for each paper 
𝑝
p

Each twin exposes three memory tiers:

Episodic Memory – granular events and artifacts.

Semantic Memory – time-structured summaries.

Persona Schema – concept graph + style attributes.

graph TD
    subgraph IndividualLevel
        RT[RepoTwin(r)] --> EPr[Episodic Memory_r]
        RT --> SMr[Semantic Memory_r]
        RT --> PSr[Persona Schema_r]

        PT[PaperTwin(p)] --> EPp[Episodic Memory_p]
        PT --> SMp[Semantic Memory_p]
        PT --> PSp[Persona Schema_p]
    end

3.1 Episodic Memory

Episodic memory stores fine-grained episodes: code chunks, text chunks, commits, paragraphs, sections.

We define a generic episode:

𝑒
=
(
id
𝑒
,
entity_id
,
time
𝑒
,
type
𝑒
,
text
𝑒
,
graph_context
𝑒
,
𝑧
𝑒
dense
,
𝑧
𝑒
sparse
)
e=(id
e
	​

,entity_id,time
e
	​

,type
e
	​

,text
e
	​

,graph_context
e
	​

,z
e
dense
	​

,z
e
sparse
	​

)

Where:

entity_id ∈ 
𝑅
∪
𝑃
R∪P (repo or paper)

time_e:

For repos: commit timestamp or approximation.

For papers: publication date or section-level surrogate.

type_e: one of:

function_def, class_def, test_case, commit_message, issue_comment, doc_paragraph (for repos)

abstract_chunk, body_chunk, equation_block, pseudo_code (for papers)

graph_context_e:

For repos: set of ProgramGraph node IDs 
⊆
𝑉
𝑟
⊆V
r
	​

 touched by this episode.

For papers: paragraphs/sections mapped to PaperGraph node(s) and concepts.

z_e_dense: embedding of text_e (code-text joint model).

z_e_sparse: sparse representation (BM25 / term-frequency).

Episodic store:

Keyed by: (entity_id, time_e, type_e)

Indexed by:

Dense vector index over 
𝑧
𝑒
dense
z
e
dense
	​


Sparse inverted index over 
𝑧
𝑒
sparse
z
e
sparse
	​


Filters on entity_id, type_e, time_range.

This implements the hybrid RAG for both RepoTwin and PaperTwin.

3.2 Semantic Memory

Semantic memory models trajectory: how a repo or paper’s ideas evolve along time or conceptual axes.

For each entity 
𝑥
∈
𝑅
∪
𝑃
x∈R∪P, we define a set of semantic summaries:

𝑠
=
(
id
𝑠
,
entity_id
=
𝑥
,
time_window
,
scope
,
summary_text
,
key_concepts
,
𝑧
𝑠
)
s=(id
s
	​

,entity_id=x,time_window,scope,summary_text,key_concepts,z
s
	​

)

time_window:

Repo: [commit_t_start, commit_t_end] or [release_v_k, release_v_{k+1}].

Paper: coarse buckets such as “preprint vs final version”, or conceptual stages if you derive them.

scope:

Subsystem or topic label (e.g., “distributed dataloader / v0.4–v0.6”, “MoE routing section”).

summary_text: LLM-generated summary over episodes in this time window + scope.

key_concepts ⊂ C: concepts emphasized in this phase.

z_s: dense embedding for semantic search.

For repositories, typical semantic summaries:

Release-level summaries: what changed and why.

Subsystem trajectories: “evolution of RL environment APIs”, “training loop refactor history”.

Bug cluster summaries: recurring failure modes and how they were fixed.

For papers, typical semantic summaries:

Conceptual pipeline: from background → method → experiments.

Relationship to prior work: what it extends/contradicts.

Evolution across related papers (if modeling author-series or follow-up work).

Semantic memory is indexed similarly to episodic memory but at a higher granularity.

3.3 Persona Schema

The Persona Schema compresses each entity’s core concept graph and stylistic profile.

For each repository 
𝑟
r:

Concept Graph 
𝐺
𝑟
persona
=
(
𝑉
𝑟
persona
,
𝐸
𝑟
persona
)
G
r
persona
	​

=(V
r
persona
	​

,E
r
persona
	​

):

Nodes 
𝑉
𝑟
persona
⊆
𝐶
V
r
persona
	​

⊆C: concepts heavily used by this repo.

Edges:

co_implemented_with: concepts frequently co-occurring within functions/classes.

refines: concept B is an implementation/detail of concept A.

tested_by: concept ↔ test patterns.

Style Attributes 
Θ
𝑟
Θ
r
	​

:

architecture_style: monolith, microservice-like, modular library, etc.

error_handling_pattern: exceptions vs return codes vs hybrid.

type_hints_density: ratio of typed vs untyped functions.

testing_style: unit-heavy vs integration-heavy, property-based usage.

performance_bias: evidence of micro-optimizations, JIT, CUDA kernels.

For each paper 
𝑝
p:

Concept Graph 
𝐺
𝑝
persona
⊆
𝐶
G
p
persona
	​

⊆C:

method concepts, theoretical constructs, empirical techniques.

Style Attributes 
Θ
𝑝
Θ
p
	​

:

theoretical_bias: purely theoretical vs empirical.

engineering_depth: degree of implementation detail.

experimentation_pattern: ablation-heavy vs single main result.

Persona schema is serialized as a promptable descriptor:

Graph: serialized adjacency, central nodes, and motifs.

Style: structured JSON that can be rendered into natural-language constraints for the LLM.

4. Domain Level: Concept and Domain Graphs

At the Domain Level, your ConceptGraph becomes the core substrate for Domain Agents that reason over topics and map them back to relevant repositories and papers.

graph TD
    subgraph DomainLevel
        DG[DomainGraph<br/>(ConceptGraph + RepoGraph + PaperGraph views)]
        DA1[DomainAgent: Deep Learning]
        DA2[DomainAgent: Compilers]
        DA3[DomainAgent: RL / Control]

        DA1 --> DG
        DA2 --> DG
        DA3 --> DG
    end

4.1 DomainGraph Structure

We define:

Concept graph 
𝐺
𝐶
=
(
𝐶
,
𝐸
𝐶
)
G
C
	​

=(C,E
C
	​

), where:

𝐸
𝐶
E
C
	​

 contains:

is_subconcept_of (taxonomic hierarchy)

co_occurs_with (in repos/papers)

appears_in_same_repo_as (induced from ProgramGraph / RepoGraph)

appears_in_same_paper_as (from PaperGraph)

Mappings:

𝑓
𝑅
:
𝑅
→
2
𝐶
f
R
	​

:R→2
C
: repo-to-concept mapping.

𝑓
𝑃
:
𝑃
→
2
𝐶
f
P
	​

:P→2
C
: paper-to-concept mapping.

Embeddings:

For each concept 
𝑐
∈
𝐶
c∈C, a vector 
𝑧
𝑐
z
c
	​

 derived from aggregated text (code docs, paper descriptions).

DomainGraph provides a unified view:

Vertices: concepts, plus (optionally) domain-level pseudo-nodes for “Deep Learning”, “Compilers”, etc.

Edges: concept relations, plus cross-domain connectors (e.g., “automatic differentiation” linking DL and compilers).

4.2 Domain Agent Tools and Variables

Each DomainAgent is an LLM with access to a set of tool APIs defined over the DomainGraph.

Key tool signatures and variables:

search_concepts(query: str) -> List[(c, score)]

Input:

query: natural language / task description.

Internals:

Compute query embedding 
𝑧
𝑞
z
q
	​

.

Return top-k concepts 
𝑐
∈
𝐶
c∈C by similarity 
sim
(
𝑧
𝑞
,
𝑧
𝑐
)
sim(z
q
	​

,z
c
	​

).

Output variables per concept:

concept_id, name, score, neighbors, top_repos, top_papers.

expand_concepts(concept_id: str) -> Neighborhood

Variables:

Direct neighbors in 
𝐸
𝐶
E
C
	​

 with edge types: is_subconcept_of, co_occurs_with, etc.

For each neighbor 
𝑐
′
c
′
:

𝑧
𝑐
′
z
c
′
	​

, local statistics (frequency in repos/papers).

find_path_between(concept_id_a, concept_id_b, k: int) -> List[Path]

Paths 
𝑃
=
(
𝑐
𝑖
1
,
…
,
𝑐
𝑖
𝐿
)
P=(c
i
1
	​

	​

,…,c
i
L
	​

	​

) with:

length, edge_types, aggregate weight (e.g., sum of inverse co-occurrence distance).

get_expert_entities_for_concept(concept_id, k) -> List[(entity_id, type, score)]

type ∈ {repo, paper}

Score from:

frequency of concept in entity,

centrality of entity within the subgraph induced by this concept.

These tools connect semantic queries to concrete repos/papers and then, via entity IDs, to RepoTwin / PaperTwin at the Individual Level.

5. Interdisciplinary Level: Orchestrator over Twins and Domains

The Interdisciplinary Level is an orchestration layer that:

Accepts tasks (from user or an upstream agent).

Detects relevant domains and concepts via DomainAgents.

Selects a set of RepoTwins and PaperTwins as experts.

Asks them to produce local plans and explanations.

Runs a review layer to integrate, de-duplicate, and check consistency.

Produces a final output: code, plan, or explanation.

sequenceDiagram
    participant U as User / Upstream Agent
    participant Coord as Coordinator
    participant Dom as DomainAgent(s)
    participant RT as RepoTwin(s)
    participant PT as PaperTwin(s)
    participant Rev as Review / Synth Agent

    U->>Coord: Task T (e.g., implement feature / analyze idea)
    Coord->>Dom: Identify domains + key concepts for T
    Dom-->>Coord: Concepts + candidate repos/papers

    loop for each selected repo r
        Coord->>RT: Subtask T_r + context
        RT-->>Coord: Repo-specific plan + evidence
    end

    loop for each selected paper p
        Coord->>PT: Subtask T_p + context
        PT-->>Coord: Paper-specific reasoning + references
    end

    Coord->>Rev: All partial plans + evidence
    Rev->>Rev: Consistency / feasibility / integration
    Rev-->>Coord: Integrated blueprint / answer
    Coord-->>U: Final code / plan / explanation

5.1 Coordinator State Variables

The Coordinator Agent maintains:

Task descriptor:

task_id, task_text, task_type ∈ {code_change, design, analysis, research_question}.

constraints: latency, target repo, style, etc.

Domain analysis:

domains: set of domain labels.

concepts: set of concepts from search_concepts.

Expert selection:

selected_repos: top-k repos from get_expert_entities_for_concept.

selected_papers: top-k papers similarly selected.

Conversation state:

Query/response history with each RepoTwin/PaperTwin.

Intermediate plans and scoring (e.g., local feasibility scores, novelty scores).

5.2 Review and Synthesis Variables

The Review / Synth Agent consumes:

Set of local plans 
{
𝜋
𝑟
}
{π
r
	​

} from repos, 
{
𝜋
𝑝
}
{π
p
	​

} from papers.

For each plan:

actions: steps or patches.

dependencies: concepts/repos/papers each step depends on.

risks: potential conflicts or gaps.

It outputs:

global_plan: ordered and merged actions.

evidence_map: mapping from action → supporting episodes (from episodic memory) + concepts.

consistency_report: any detected contradictions or ambiguity.

uncertainty_flags: where human review or extra experiments are recommended.

6. End-to-End Context Assembly

The core operation of this architecture is context assembly: building the exact prompt context for each lower-level LLM call.

Below is a canonical RepoTwin context assembly pipeline.

flowchart LR
    T[Subtask T_r] --> P1[1. Persona Loading]
    P1 --> P2[2. Semantic Scoping]
    P2 --> P3[3. Episodic Retrieval]
    P3 --> P4[4. Prompt Assembly]
    P4 --> LLM[RepoTwin LLM Call]

    subgraph Inputs
        RId[repo_id]
        Graph[ProgramGraph_r + PersonaSchema_r]
        Sem[SemanticMemory_r]
        Epis[EpisodicMemory_r]
    end

    RId --> P1
    Graph --> P1
    Sem --> P2
    Epis --> P3


Variables per stage:

Persona Loading

Input:

repo_id, PersonaSchema_r (concept graph + style attributes).

Output:

persona_prompt_r: text template describing how this project usually organizes code, patterns to prefer/avoid.

Semantic Scoping

Input:

task_text, SemanticMemory_r.

Operation:

Embed task_text and query semantic summaries:

filter by time window (if specified),

rank by similarity.

Output:

semantic_context_r: top-k summary objects plus their key_concepts.

Episodic Retrieval

Input:

task_text, semantic_context_r, EpisodicMemory_r.

Operation:

Use key_concepts and candidate subsystems to:

restrict to relevant ProgramGraph nodes and episodes,

run dense + sparse retrieval,

apply scoring to prefer episodes that:

are recent,

highly central,

appear in tests and docs.

Output:

episodic_context_r: top-k code/functions/tests/issues.

Prompt Assembly

Compose:

system: persona_prompt_r + high-level constraints.

context: semantic_context_r + episodic_context_r (minimized for length and redundancy).

user: the subtask T_r.

A similar pipeline is used for PaperTwin, replacing ProgramGraph constraints with paper structure and citation context.

7. Summary

By mapping MirrorMind onto your existing repository and paper graphs, you obtain:

Individual Level

RepoTwin and PaperTwin agents with:

Episodic memory over fine-grained code/text/commit/paragraph episodes.

Semantic memory over release-level and concept-level trajectories.

Persona schemas as compact concept graphs and style profiles.

Domain Level

DomainAgents operating over a unified ConceptGraph/DomainGraph:

Tools for concept search, expansion, path-finding, and expert entity retrieval.

Variables describing concept embeddings, co-occurrence statistics, and entity mappings.

Interdisciplinary Level

A Coordinator and Review/Synthesis layer that:

Decomposes tasks, selects relevant repos/papers, and queries their twins.

Integrates local plans and evidence into coherent global blueprints.

All of this is implemented without changing your underlying graphs; the architecture is a view layer that defines the variables, memory structures, and agent APIs over your existing ProgramGraph, RepoGraph, PaperGraph, and ConceptGraph.

8. Implementation Status in This Repository

This repository provides a concrete, minimal instantiation of the above architecture, with a focus on making the variables and data structures from Sections 3–6 directly inspectable.

8.1 Implemented Components

Episodic memory:

- `models/mirrormind/memory.py` defines `Episode` and `EpisodicMemoryStore` with:
- `entity_id ∈ R ∪ P`, `time_e`, `type_e`, `text_e`, `graph_context_e`, `z_e_dense`, and placeholder `z_e_sparse`,
- hybrid retrieval over dense/sparse indices plus a recency-aware heuristic.
- Episodic stores are currently populated from:
- repository code/documentation chunks via `models/mirrormind/scripts/build_semantic_from_repo_chunks.py`,
- lightweight paper manifests in `PaperTwin` (titles, abstracts as episodes).

Semantic memory:

- `SemanticSummary` and `SemanticMemoryStore` (also in `memory.py`) implement the trajectory-level variables from Section 3.2:
- `time_window`, `scope`, `summary_text`, `key_concepts`, `z_s`.
- `build_semantic_summaries` aggregates episodes per-entity into coarse semantic windows.
- `build_semantic_from_repo_chunks.py` uses an LLM summarizer to write semantic summaries to `models/exports/semantic_from_chunks.jsonl` with `scope="repo_chunks"`.

Persona schemas:

- `models/mirrormind/persona.py` defines `PersonaSchema` and `PersonaBuilder`:
- `concepts_r` is derived from `models/exports/repo_concepts.jsonl`,
- edges are simple `co_implemented_with` co-occurrence links over consecutive concepts,
- style attributes `Θ_r` include `architecture_style`, `testing_style`, `performance_bias`, `type_hints_density`, plus a few optional heuristics.
- `models/mirrormind/twins.py` wires these into `RepoTwin` and `PaperTwin`, exposing `persona_prompt_r` / `persona_prompt_p`.

Domain graph and domain agent:

- `models/mirrormind/domain.py` implements `ConceptNode`, `DomainGraph`, and `DomainAgent`:
- concept embeddings `z_c` via `TextEmbedder`,
- edges `E_C` currently include `co_occurs_with`, `appears_in_same_repo_as`, and `appears_in_same_paper_as`,
- repo/paper mappings `f_R`, `f_P` via `top_repos`, `top_papers`,
- tool APIs `search_concepts`, `expand_concepts`, `find_path_between`, and `get_expert_entities_for_concept`.
- `models/mirrormind/graph_client.py` provides a file-backed client used by tests and by the default `Coordinator`.

Coordinator and review layer:

- `models/mirrormind/coordinator.py` implements:
- `TaskDescriptor` with `task_id`, `task_text`, `task_type`, and `constraints`,
- `Coordinator.run` and `run_multi_step`, which:
- call `DomainAgent.search_concepts`,
- select expert repositories/papers (`selected_repos`, `selected_papers`) via `get_expert_entities_for_concept`,
- spawn `RepoTwin` / `PaperTwin` instances and request local plans,
- aggregate results and pass them to `ReviewAgent`.
- `ReviewAgent.synthesize` implements the Review/Synth variables of Section 5.2 in a lightweight form:
- `global_plan` (scored, de-duplicated actions),
- `evidence_map` and `evidence_summary`,
- `dependencies`,
- `consistency_report`,
- `issues` and `uncertainty_flags` derived from heuristic pattern checks.

Context assembly:

- `models/mirrormind/context.py` provides a concrete context assembly pipeline (Section 6) via `ContextAssembler`:
- calls `persona_prompt`, `semantic_scope`, and `episodic_context` on each twin,
- de-duplicates and truncates semantic/episodic items to fit within prompt budgets,
- emits `{system, semantic_context, episodic_context, user_task}` blocks used by downstream LLM calls (e.g., via `PianoAgent`).

8.2 Known Simplifications and Open Gaps

The current implementation intentionally leaves several aspects of the full spec as stubs or simplifications:

- Temporal/evolution modeling:
- commits `K_r`, diff summaries, and ΔV_r,t are not explicitly tracked; `Episode.time` and `SemanticSummary.time_window` are present but only lightly used.
- the `repo_chunks` import path assigns `time=None`, so those summaries do not yet carry structured time windows or release scopes.

- Semantic scoping and episodic retrieval:
- semantic scoping is implemented inside `SemanticMemoryStore.query` (embedding + token overlap),
- episodic retrieval in `EpisodicMemoryStore.query` uses hybrid dense/sparse indices with recency and type-aware weighting,
- but there is no ProgramGraph-based restriction to subsystems and no second-stage semantic scoping layer inside `ContextAssembler`.

- Persona richness:
- persona schemas are based on exported concepts and simple heuristics (tests, CUDA, type hints, doc tokens),
- paper personas reuse the same machinery with shuffled concepts,
- there is no learned extraction of architecture/error-handling/testing patterns from the underlying graphs or paper structure.

- Domain graph taxonomy:
- `DomainGraph` currently models co-occurrence edges but does not include a full `is_subconcept_of` hierarchy, domain-level pseudo-nodes, or explicit cross-domain connectors.

- Review/synthesis depth:
- `ReviewAgent` provides heuristic conflict and uncertainty detection but does not yet:
- attribute evidence back to specific episodes/semantic windows,
- model risk or feasibility scores beyond simple pattern-based downgrades,
- merge overlapping actions using graph-level reasoning.

- Context optimization:
- the context assembler performs de-duplication and truncation only;
- it does not yet incorporate graph-aware filtering (e.g., via `DomainGraph` paths or ProgramGraph neighborhoods) when choosing which summaries/episodes to surface.

- Cross-graph integration and data coverage:
- repo and paper concepts are linked at the concept level and via optional paper–repo alignment files,
- but there are no explicit RepoGraph/PaperGraph structures or citation edges,
- paper parsing is limited to metadata (titles/abstracts), and repository issues/PR discussions are not yet ingested as episodes.

These gaps are deliberate: they keep the current codebase compact while making the specification in this document concrete. Future iterations can extend the existing classes and exports to close each gap without changing the overall variable definitions.