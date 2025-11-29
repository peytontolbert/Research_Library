# Gap Analysis: MirrorMind Paper vs Implementation

This document identifies gaps between the MirrorMind architecture specification in `paper.md` and the current implementation in `models/mirrormind/`.

## Summary

The implementation provides a **working scaffold** with basic structures in place, but several critical components are **stubby, heuristic, or missing**. The gaps fall into these categories:

1. **Temporal/Evolution Tracking** - Missing proper time-window handling
2. **Concept Graph Structure** - Missing typed edges and proper taxonomy
3. **Review/Synthesis Layer** - Very basic, missing consistency checking
4. **Semantic Memory Generation** - No LLM-based summary generation
5. **Persona Schema Extraction** - Heuristic-based, not learned
6. **Cross-Graph Integration** - Limited paper-repo alignment
7. **Context Assembly Optimization** - No redundancy minimization
8. **Variable Completeness** - Several paper-specified variables missing

---

## 1. Episodic Memory Gaps

### ✅ Implemented
- Basic episode structure (`id`, `entity_id`, `time`, `type`, `text`, `graph_context`, `dense`, `sparse`)
- Hybrid dense/sparse indexing (FAISS/BM25)
- Query with filters (entity_id, types, time_range)

### ❌ Missing/Incomplete

**1.1 Temporal Granularity**
- **Paper spec (Section 3.1)**: `time_e` should be commit timestamp for repos, publication date for papers
- **Current**: `time` is optional string, no parsing/validation
- **Gap**: No structured time handling (commit sequences `K_r`, timestamps `t(k_t)`)
- **Impact**: Can't properly filter by time windows or track evolution

**1.2 Episode Type Coverage**
- **Paper spec**: Full type taxonomy:
  - Repos: `function_def`, `class_def`, `test_case`, `commit_message`, `issue_comment`, `doc_paragraph`
  - Papers: `abstract_chunk`, `body_chunk`, `equation_block`, `pseudo_code`
- **Current**: Types are strings, no validation/enumeration
- **Gap**: No structured episode type system or type-specific handling

**1.3 Graph Context Mapping**
- **Paper spec**: `graph_context_e` should map to ProgramGraph node IDs `⊆ V_r` for repos
- **Current**: `graph_context` is `Sequence[str]` with no validation
- **Gap**: No verification that context IDs exist in ProgramGraph, no bidirectional mapping

**1.4 Sparse Representation Quality**
- **Paper spec**: `z_e_sparse` should be BM25/term-frequency based
- **Current**: Sparse index exists but uses simple text matching
- **Gap**: No proper BM25 weighting or term-frequency normalization

---

## 2. Semantic Memory Gaps

### ✅ Implemented
- Basic `SemanticSummary` structure
- Simple similarity scoring (token overlap + dense magnitude)
- Query by entity_id, scope, text

### ❌ Missing/Incomplete

**2.1 Time Window Structure**
- **Paper spec (Section 3.2)**: 
  - Repos: `[commit_t_start, commit_t_end]` or `[release_v_k, release_v_{k+1}]`
  - Papers: "preprint vs final version" or conceptual stages
- **Current**: `time_window` is a string (e.g., `entry.get("update_date", "")`)
- **Gap**: No structured time intervals, no release version tracking, no commit range parsing

**2.2 Scope Granularity**
- **Paper spec**: Subsystem/topic labels like "distributed dataloader / v0.4–v0.6", "MoE routing section"
- **Current**: `scope` is a simple string (e.g., `"abstract"`)
- **Gap**: No structured scope taxonomy, no version-aware scoping

**2.3 Summary Generation**
- **Paper spec (Section 3.2)**: `summary_text` should be "LLM-generated summary over episodes in this time window + scope"
- **Current**: Summaries are manually created or copied from abstracts
- **Gap**: **No automatic summary generation pipeline** - this is a major missing feature

**2.4 Semantic Summary Types**
- **Paper spec**: Specific summary types:
  - Release-level summaries
  - Subsystem trajectories ("evolution of RL environment APIs")
  - Bug cluster summaries
- **Current**: Generic summaries only
- **Gap**: No specialized summary generation for different trajectory types

**2.5 Key Concepts Extraction**
- **Paper spec**: `key_concepts ⊂ C` (concepts emphasized in this phase)
- **Current**: `key_concepts` extracted from categories (simple split)
- **Gap**: No proper concept extraction from summary text, no validation against ConceptGraph

---

## 3. Persona Schema Gaps

### ✅ Implemented
- Basic `PersonaSchema` structure (concepts, edges, style)
- Prompt rendering
- Concept loading from exports

### ❌ Missing/Incomplete

**3.1 Concept Graph Structure**
- **Paper spec (Section 3.3)**: 
  - Nodes `V_r_persona ⊆ C` (concepts heavily used by repo)
  - Edges: `co_implemented_with`, `refines`, `tested_by`
- **Current**: 
  - Concepts loaded from exports (good)
  - Edges are simple consecutive co-occurrence (`co_implemented_with` only)
- **Gap**: 
  - No `refines` (concept B is implementation/detail of A)
  - No `tested_by` (concept ↔ test patterns)
  - No proper edge type taxonomy

**3.2 Style Attribute Extraction**
- **Paper spec**: Comprehensive style attributes:
  - `architecture_style`: monolith, microservice-like, modular library
  - `error_handling_pattern`: exceptions vs return codes vs hybrid
  - `type_hints_density`: ratio of typed vs untyped functions
  - `testing_style`: unit-heavy vs integration-heavy, property-based
  - `performance_bias`: evidence of micro-optimizations, JIT, CUDA kernels
- **Current**: Heuristic-based (`_infer_style`) with simple keyword matching
- **Gap**: 
  - No proper analysis of code patterns
  - No ratio calculations (e.g., typed vs untyped)
  - No evidence-based detection (e.g., CUDA kernel usage)

**3.3 Paper Persona Schema**
- **Paper spec**: For papers:
  - Concept graph with method concepts, theoretical constructs, empirical techniques
  - Style: `theoretical_bias`, `engineering_depth`, `experimentation_pattern`
- **Current**: Paper persona uses same builder with `persona_type="paper"` but no paper-specific logic
- **Gap**: No paper-specific concept extraction or style attributes

**3.4 Persona Serialization**
- **Paper spec**: "Graph: serialized adjacency, central nodes, and motifs"
- **Current**: Simple text prompt
- **Gap**: No structured serialization format (JSON/GraphML), no motif detection

---

## 4. Domain Level Gaps

### ✅ Implemented
- Basic `DomainGraph` structure
- Concept search, expansion, path-finding
- Expert entity retrieval
- DomainAgent tool APIs

### ❌ Missing/Incomplete

**4.1 Concept Graph Edge Types**
- **Paper spec (Section 4.1)**: `E_C` should contain:
  - `is_subconcept_of` (taxonomic hierarchy)
  - `co_occurs_with` (in repos/papers)
  - `appears_in_same_repo_as` (from ProgramGraph/RepoGraph)
  - `appears_in_same_paper_as` (from PaperGraph)
- **Current**: Only lightweight neighbor links (concepts sharing a repo_id)
- **Gap**: 
  - No `is_subconcept_of` taxonomy
  - No explicit `co_occurs_with` edges
  - No proper edge type system

**4.2 Concept Embeddings**
- **Paper spec**: "For each concept c ∈ C, a vector z_c derived from aggregated text (code docs, paper descriptions)"
- **Current**: `node.embedding` exists but is empty list by default
- **Gap**: **No embedding computation for concepts** - this is critical for semantic search

**4.3 Domain-Level Pseudo-Nodes**
- **Paper spec**: "optionally domain-level pseudo-nodes for 'Deep Learning', 'Compilers', etc."
- **Current**: No domain-level nodes
- **Gap**: No high-level domain abstraction layer

**4.4 Cross-Domain Connectors**
- **Paper spec**: "cross-domain connectors (e.g., 'automatic differentiation' linking DL and compilers)"
- **Current**: No explicit cross-domain edge types
- **Gap**: No mechanism to identify and represent cross-domain relationships

**4.5 Expert Entity Scoring**
- **Paper spec (Section 4.2)**: Score from:
  - Frequency of concept in entity
  - Centrality of entity within subgraph induced by this concept
- **Current**: Simple random shuffle with decreasing score
- **Gap**: No frequency calculation, no centrality metrics

---

## 5. Interdisciplinary Level (Coordinator) Gaps

### ✅ Implemented
- Basic `Coordinator` structure
- Task routing through DomainAgent
- Repo/paper selection
- ReviewAgent stub

### ❌ Missing/Incomplete

**5.1 Task Descriptor Completeness**
- **Paper spec (Section 5.1)**: Full task descriptor:
  - `task_id`, `task_text`, `task_type ∈ {code_change, design, analysis, research_question}`
  - `constraints`: latency, target repo, style, etc.
- **Current**: Basic `TaskDescriptor` with `task_id`, `task_text`, `task_type`, `constraints` dict
- **Gap**: No validation of `task_type` enum, no structured constraint parsing

**5.2 Domain Analysis Variables**
- **Paper spec**: Coordinator should maintain:
  - `domains`: set of domain labels
  - `concepts`: set of concepts from `search_concepts`
- **Current**: Concepts searched but not stored in coordinator state
- **Gap**: No persistent domain/concept state tracking

**5.3 Expert Selection Refinement**
- **Paper spec**: "top-k repos from `get_expert_entities_for_concept`"
- **Current**: Uses `search_concepts` then extracts repos from results
- **Gap**: Should use `get_expert_entities_for_concept` directly for better scoring

**5.4 Conversation State**
- **Paper spec**: "Query/response history with each RepoTwin/PaperTwin"
- **Current**: No conversation history tracking
- **Gap**: No stateful interaction with twins

**5.5 Intermediate Plans and Scoring**
- **Paper spec**: "local feasibility scores, novelty scores"
- **Current**: Plans are simple string actions
- **Gap**: No scoring mechanism, no feasibility analysis

---

## 6. Review and Synthesis Gaps

### ✅ Implemented
- Basic `ReviewAgent.synthesize()` method
- Action deduplication
- Simple issue detection

### ❌ Missing/Incomplete

**6.1 Plan Structure**
- **Paper spec (Section 5.2)**: Each plan should have:
  - `actions`: steps or patches
  - `dependencies`: concepts/repos/papers each step depends on
  - `risks`: potential conflicts or gaps
- **Current**: Plans are dicts with `actions` and `evidence`, no structured dependencies/risks
- **Gap**: No dependency tracking, no risk analysis

**6.2 Consistency Checking**
- **Paper spec**: "consistency_report: any detected contradictions or ambiguity"
- **Current**: Returns `"stub-consistent"` or `"no-actions"`
- **Gap**: **No actual consistency checking logic** - this is a major missing feature

**6.3 Evidence Mapping**
- **Paper spec**: "evidence_map: mapping from action → supporting episodes (from episodic memory) + concepts"
- **Current**: Simple index-based evidence map
- **Gap**: No linking to actual episodes, no concept attribution

**6.4 Uncertainty Flags**
- **Paper spec**: "uncertainty_flags: where human review or extra experiments are recommended"
- **Current**: Not implemented
- **Gap**: **Missing feature** - no uncertainty detection

**6.5 Integration Quality**
- **Paper spec**: Should "integrate, de-duplicate, and check consistency"
- **Current**: Basic deduplication, no integration logic
- **Gap**: No merging of similar actions, no conflict resolution

---

## 7. Context Assembly Gaps

### ✅ Implemented
- Basic context assembly pipeline
- Persona + semantic + episodic + task composition
- Formatting helpers

### ❌ Missing/Incomplete

**7.1 Prompt Optimization**
- **Paper spec (Section 6)**: "minimized for length and redundancy"
- **Current**: No minimization, just concatenation
- **Gap**: **No redundancy removal, no length optimization** - critical for token limits

**7.2 Semantic Scoping with Embeddings**
- **Paper spec**: "Embed task_text and query semantic summaries: filter by time window (if specified), rank by similarity"
- **Current**: Uses text overlap, no embedding-based similarity
- **Gap**: Should use dense embeddings for semantic search

**7.3 Episodic Retrieval Scoring**
- **Paper spec**: Prefer episodes that are:
  - Recent
  - Highly central
  - Appear in tests and docs
- **Current**: Basic scoring (dense_score * 0.6 + overlap * 0.4)
- **Gap**: No recency weighting, no centrality calculation, no test/doc preference

**7.4 Graph Context Filtering**
- **Paper spec**: "restrict to relevant ProgramGraph nodes and episodes"
- **Current**: No ProgramGraph integration in context assembly
- **Gap**: Should filter episodes by graph node relevance

---

## 8. Cross-Graph Integration Gaps

### ✅ Implemented
- Basic paper-repo alignment loading from exports
- Concept nodes can have both repos and papers

### ❌ Missing/Incomplete

**8.1 Paper Graph Integration**
- **Paper spec**: Assumes `PaperGraph` exists with citations, topical links
- **Current**: Paper concepts loaded from JSONL, no graph structure
- **Gap**: No proper PaperGraph implementation, no citation tracking

**8.2 Repo Graph Integration**
- **Paper spec**: Assumes `RepoGraph` exists (repositories as nodes, similarity edges)
- **Current**: Repo concepts loaded, but no RepoGraph structure
- **Gap**: No repository-level similarity graph

**8.3 Concept Graph Unification**
- **Paper spec**: "ConceptGraph (concepts unified across code and papers, explicit or implicit)"
- **Current**: Concepts loaded separately, no unified view
- **Gap**: No single unified ConceptGraph abstraction

**8.4 Cross-Graph Mappings**
- **Paper spec**: 
  - `f_R: R → 2^C` (repo-to-concept mapping)
  - `f_P: P → 2^C` (paper-to-concept mapping)
- **Current**: Mappings exist implicitly in concept nodes
- **Gap**: No explicit mapping functions, no inverse lookups

---

## 9. Data Structure Completeness

### Missing Variables from Paper

**9.1 Repository Variables (Section 2.1)**
- ✅ `id_r`, `name_r`, `url_r` - in manifest
- ✅ `languages_r`, `tags_r` - in manifest
- ✅ `stars_r`, `forks_r`, `size_r` - optional, not always present
- ✅ Program graph `V_r`, `E_r` - via ProgramGraph
- ❌ **Commit sequence `K_r`** - not tracked
- ❌ **Commit timestamps `t(k_t)`** - not tracked
- ❌ **Diff summaries** - not tracked
- ❌ **Changed nodes `ΔV_r,t`** - not tracked
- ✅ Textual artifacts - partial (README, docs)
- ❌ **Issue bodies, PR discussions** - not tracked
- ✅ `linked_papers_r` - via concept graph
- ✅ `concepts_r` - via concept graph

**9.2 Paper Variables (Section 2.2)**
- ✅ `id_p`, `doi_p` - in manifest
- ✅ `title_p`, `venue_p` - in manifest
- ✅ `year_p` - in manifest
- ✅ `authors_p` - in manifest
- ✅ `fields_p` - via categories
- ✅ Textual content - abstract available
- ❌ **Full sections, equations, pseudo-code** - not parsed from PDFs
- ❌ **Citations `cites_p`** - not tracked
- ❌ **Cited-by edges** - not tracked
- ✅ `linked_repos_p` - via concept graph
- ✅ `concepts_p` - via concept graph

---

## 10. Implementation Quality Gaps

### 10.1 Error Handling
- Many functions have basic try/except but don't surface meaningful errors
- No validation of input data structures
- No schema validation for episodes/summaries

### 10.2 Persistence
- JSONL save/load exists but no versioning
- No migration path for schema changes
- No incremental updates (always full save/load)

### 10.3 Performance
- No caching of expensive operations (embeddings, graph traversals)
- No batch processing optimizations
- Index building is synchronous

### 10.4 Testing
- Limited test coverage
- No integration tests for end-to-end flows
- No performance benchmarks

---

## Priority Recommendations

### Critical (Blocks Core Functionality)
1. **Semantic Summary Generation** - Implement LLM-based summary generation pipeline
2. **Concept Embeddings** - Compute embeddings for all concepts in DomainGraph
3. **Temporal Handling** - Proper time window parsing and commit sequence tracking
4. **Consistency Checking** - Implement actual consistency detection in ReviewAgent

### High Priority (Significant Value)
5. **Context Optimization** - Implement redundancy removal and length minimization
6. **Persona Style Extraction** - Replace heuristics with proper code analysis
7. **Edge Type System** - Implement full edge taxonomy (is_subconcept_of, refines, etc.)
8. **Expert Scoring** - Implement frequency and centrality-based scoring

### Medium Priority (Nice to Have)
9. **Uncertainty Flags** - Add uncertainty detection to ReviewAgent
10. **Conversation State** - Track query/response history
11. **Paper-Specific Persona** - Implement paper-specific concept/style extraction
12. **Cross-Domain Connectors** - Identify and represent cross-domain relationships

---

## Notes

- The implementation is a **good scaffold** - the structure is correct, but many components need to move from "heuristic/stub" to "proper implementation"
- Many gaps are **data-dependent** - they require proper graph construction and concept extraction pipelines
- Some gaps are **model-dependent** - they require trained models (e.g., summary generation, style extraction)
- The paper assumes certain infrastructure (ProgramGraph, RepoGraph, PaperGraph) that may need to be built first

