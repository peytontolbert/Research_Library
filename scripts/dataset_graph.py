from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Dict, Optional

from modules.program_graph import (
	ProgramGraph,
	Entity,
	Edge,
	Artifact,
	Span,
	ResolvedAnchor,
	EntityId,
)
from .repo_graph import artifact_uri, program_id_for_repo, parse_program_uri


class DatasetProgramGraph(ProgramGraph):
	"""
	Minimal Dataset→ProgramGraph backend.
	Represents benchmark datasets (MBPP, SWE-bench, etc.) as entities:
	- problem:<dataset>/<split>/<id>
	- reference:<dataset>/<split>/<id>
	- verifier:<dataset>/<split>/<id>
	Artifacts point to source JSON or local cache paths when available.
	"""

	def __init__(self, root: str, datasets: Optional[List[str]] = None):
		self.root = os.path.abspath(root)
		self.program_id = program_id_for_repo(self.root)
		self._datasets = datasets or ["mbpp:train", "mbpp:test", "swe_bench:train"]
		# Tiny synthetic index for smoke use
		self._index: Dict[str, Dict[str, str]] = {}
		for spec in self._datasets:
			ds, split = (spec.split(":", 1) + ["train"])[:2]
			key = f"{ds}/{split}/0001"
			self._index[key] = {
				"dataset": ds,
				"split": split,
				"problem": f"Solve sample problem 0001 from {ds}/{split}",
				"reference": "def solution(x):\n\treturn x\n",
				"verifier": "python verify_0001.py",
				"artifact_rel": f"datasets/{ds}/{split}/0001.json"
			}

	def entities(self) -> Iterable[Entity]:
		out: List[Entity] = []
		for key, meta in self._index.items():
			ds = meta["dataset"]
			split = meta["split"]
			# problem
			out.append(Entity(
				uri=f"program://{self.program_id}/problem/{key}",
				id=f"ds:problem:{key}",
				kind="problem",
				name=f"{ds}:{split}:{key.split('/')[-1]}",
				owner=None,
				labels=None,
			))
			# reference
			out.append(Entity(
				uri=f"program://{self.program_id}/reference/{key}",
				id=f"ds:reference:{key}",
				kind="reference",
				name=f"{ds}:{split}:{key.split('/')[-1]}",
				owner=None,
				labels=None,
			))
			# verifier
			out.append(Entity(
				uri=f"program://{self.program_id}/verifier/{key}",
				id=f"ds:verifier:{key}",
				kind="verifier",
				name=f"{ds}:{split}:{key.split('/')[-1]}",
				owner=None,
				labels=None,
			))
		return out

	def edges(self) -> Iterable[Edge]:
		out: List[Edge] = []
		for key in self._index.keys():
			pid = f"ds:problem:{key}"
			rid = f"ds:reference:{key}"
			vid = f"ds:verifier:{key}"
			out.append(Edge(src=pid, dst=rid, type="has_reference", meta=None))
			out.append(Edge(src=pid, dst=vid, type="has_verifier", meta=None))
		return out

	def artifacts(self, kind: str) -> Iterable[Artifact]:
		if kind not in ("artifact", "source"):
			return []
		out: List[Artifact] = []
		for key, meta in self._index.items():
			rel = meta.get("artifact_rel") or f"datasets/{key}.json"
			abs_fp = os.path.join(self.root, rel)
			# We may not have real files; just issue URIs with empty hash
			out.append(Artifact(uri=artifact_uri(self.program_id, rel), type="artifact", hash="", span=None))
		return out

	def search_refs(self, token: str) -> Iterable[Tuple[EntityId, Span]]:
		t = (token or "").strip().lower()
		if not t:
			return []
		out: List[Tuple[EntityId, Span]] = []
		for key, meta in self._index.items():
			if t in (meta.get("problem") or "").lower():
				out.append((f"ds:problem:{key}", Span(start_line=1, end_line=1)))
		return out

	def resolve(self, uri: str) -> ResolvedAnchor:
		pid, kind, res, span = parse_program_uri(uri)
		if pid != self.program_id:
			raise ValueError(f"program id mismatch: {pid} != {self.program_id}")
		# Map entity URIs to synthetic artifact anchors
		if kind in ("problem", "reference", "verifier"):
			key = res
			base = f"datasets/{key}.json"
			a = int(span[0]) if span else 1
			b = int(span[1]) if span else a
			return ResolvedAnchor(artifact_uri=artifact_uri(self.program_id, base), span=Span(start_line=a, end_line=b), hash="")
		# Fallback to artifact resolution
		if kind == "artifact":
			# Defer to generic artifact handling through RepoGraph helpers (inline)
			abs_fp = os.path.abspath(os.path.join(self.root, res))
			a = int(span[0]) if span else 1
			b = int(span[1]) if span else max(1, a)
			rel = res.replace("\\", "/")
			return ResolvedAnchor(artifact_uri=artifact_uri(self.program_id, rel), span=Span(start_line=a, end_line=b), hash="")
		raise KeyError(f"unrecognized entity uri for kind={kind}, resource={res}")


