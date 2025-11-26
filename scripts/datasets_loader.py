from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable


def _read_json(path: Path) -> Dict[str, Any]:
	try:
		with open(path, "r", encoding="utf-8") as fh:
			return json.loads(fh.read())
	except Exception:
		return {}


def _load_local_jsonl(fp: Path, text_key: str = "text", max_n: int | None = None) -> List[str]:
	texts: List[str] = []
	if not fp.exists():
		return texts
	with open(fp, "r", encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			txt = str(obj.get(text_key) or "").strip()
			if txt:
				texts.append(txt)
			if max_n is not None and len(texts) >= int(max_n):
				break
	return texts


def _load_hf(
	repo: str,
	subset: str | None,
	split: str,
	max_n: int | None,
	cache_dir: Path,
	streaming: bool,
	prompt_key: str | None = None,
	code_key: str | None = None,
	compose: List[str] | None = None,
	join_with: str = "\n",
) -> List[str]:
	try:
		from datasets import load_dataset  # type: ignore
	except Exception:
		return []
	try:
		if subset:
			ds = load_dataset(repo, subset, split=split, cache_dir=str(cache_dir), streaming=streaming)
		else:
			ds = load_dataset(repo, split=split, cache_dir=str(cache_dir), streaming=streaming)
	except Exception:
		return []
	texts: List[str] = []
	# Normalize and compose a row into text based on provided keys or sensible defaults
	def _normalize_value(v: Any) -> str:
		if isinstance(v, (list, tuple)):
			if not v:
				return ""
			v = v[0]
		if v is None:
			return ""
		try:
			s = str(v).strip()
		except Exception:
			s = ""
		return s
	def _compose_row(row: Dict[str, Any]) -> str:
		# Highest priority: explicit compose list
		if compose and isinstance(compose, list):
			parts: List[str] = []
			for k in compose:
				parts.append(_normalize_value(row.get(k)))
			joined = join_with.join([p for p in parts if p])
			return joined.strip()
		# Next: explicit prompt/code keys
		if prompt_key or code_key:
			prompt = _normalize_value(row.get(prompt_key or ""))
			code = _normalize_value(row.get(code_key or ""))
			return (prompt + ((join_with + code) if code else "")).strip()
		# Fallback: best-effort defaults
		prompt = _normalize_value(row.get("prompt") or row.get("question") or row.get("text"))
		code = _normalize_value(row.get("code") or row.get("solution"))
		return (prompt + ((join_with + code) if code else "")).strip()
	if streaming:
		count = 0
		for row in ds:  # type: ignore
			combined = _compose_row(row)  # type: ignore
			if combined:
				texts.append(combined)
				count += 1
			if max_n is not None and count >= int(max_n):
				break
		return texts
	# non-streaming
	total = len(ds)  # type: ignore
	for i in range(total):
		row = ds[i]  # type: ignore
		combined = _compose_row(row)  # type: ignore
		if combined:
			texts.append(combined)
		if max_n is not None and len(texts) >= int(max_n):
			break
	return texts


def load_program_texts(example_dir: str, config_path: str | None = None, train_new_only: bool = False, state_path: str | None = None) -> Tuple[List[str], List[str]]:
	"""
	Load training texts from a config that can reference local and HF datasets.
	Priority: local first; if missing, fetch from HF (optionally streaming) and store under local_dir.

	Returns (texts, loaded_source_names)
	"""
	ex_dir = Path(example_dir).resolve()
	# Config discovery
	cfg_path = Path(config_path) if config_path else (ex_dir / "datasets" / "config.json")
	cfg = _read_json(cfg_path) if cfg_path.exists() else {}
	local_dir = Path(cfg.get("local_dir") or (ex_dir / "datasets")).resolve()
	local_dir.mkdir(parents=True, exist_ok=True)
	cache_dir = local_dir / "hf_cache"
	cache_dir.mkdir(parents=True, exist_ok=True)
	sources: List[Dict[str, Any]] = list(cfg.get("sources") or [])
	streaming = os.getenv("HF_DATASETS_STREAMING", "1") == "1"
	full_datasets = os.getenv("FULL_DATASETS", "1") == "1"
	# Empty/0 -> unlimited by default
	_default_env = os.getenv("DATASET_MAX_N_PER_SOURCE", "") or "0"
	try:
		default_max_n_val = int(_default_env)
	except Exception:
		default_max_n_val = 0

	# Optional program state to support "train new only"
	seen: List[str] = []
	if train_new_only and state_path:
		try:
			st = _read_json(Path(state_path))
			seen = list(st.get("datasets_seen") or [])
		except Exception:
			seen = []

	texts_acc: List[str] = []
	loaded_names: List[str] = []
	for src in sources:
		name = str(src.get("name") or "").strip()
		if not name:
			continue
		if train_new_only and name in seen:
			continue
		if full_datasets:
			max_n = None
		else:
			max_n = src.get("max_n")
			if max_n is None and default_max_n_val > 0:
				max_n = default_max_n_val
		text_key = str(src.get("text_key") or "text").strip()
		subset = src.get("subset")
		split = str(src.get("split") or "train").strip()
		# Determine local file path: explicit 'path' or derived from name+split
		rel = str(src.get("path") or "").strip()
		if rel:
			fp = Path(rel)
			if not fp.is_absolute():
				fp = (local_dir / rel).resolve()
		else:
			suffix = f"_{split}" if split else ""
			fp = (local_dir / f"{name.replace('/', '_')}{suffix}.jsonl").resolve()
		# Try local first
		out_texts: List[str] = _load_local_jsonl(fp, text_key=text_key, max_n=max_n)
		# If not present, try HF using 'name' as repo id
		if not out_texts:
			out_texts = _load_hf(
				name,
				subset,
				split,
				max_n,
				cache_dir,
				streaming=streaming,
				prompt_key=(src.get("prompt_key") if isinstance(src.get("prompt_key"), str) else None),
				code_key=(src.get("code_key") if isinstance(src.get("code_key"), str) else None),
				compose=(src.get("compose") if isinstance(src.get("compose"), list) else None),
				join_with=str(src.get("join_with") or "\n"),
			)
			# Persist to local for future runs
			if out_texts:
				try:
					with open(fp, "w", encoding="utf-8") as fh:
						for t in out_texts:
							fh.write(json.dumps({"text": t}) + "\n")
				except Exception:
					pass
		if out_texts:
			texts_acc.extend(out_texts)
			loaded_names.append(f"{name}:{split}" if split else name)
	return texts_acc, loaded_names


def iter_program_texts(example_dir: str, config_path: str | None = None, train_new_only: bool = False, state_path: str | None = None) -> Iterable[Tuple[List[str], str]]:
	"""
	Yields (texts, source_name) one dataset at a time using the same resolution rules as load_program_texts.
	Respects HF streaming and per-source max limits.
	"""
	ex_dir = Path(example_dir).resolve()
	cfg_path = Path(config_path) if config_path else (ex_dir / "datasets" / "config.json")
	cfg = _read_json(cfg_path) if cfg_path.exists() else {}
	local_dir = Path(cfg.get("local_dir") or (ex_dir / "datasets")).resolve()
	local_dir.mkdir(parents=True, exist_ok=True)
	cache_dir = local_dir / "hf_cache"
	cache_dir.mkdir(parents=True, exist_ok=True)
	sources: List[Dict[str, Any]] = list(cfg.get("sources") or [])
	streaming = os.getenv("HF_DATASETS_STREAMING", "1") == "1"
	full_datasets = os.getenv("FULL_DATASETS", "1") == "1"
	_default_env = os.getenv("DATASET_MAX_N_PER_SOURCE", "") or "0"
	try:
		default_max_n_val = int(_default_env)
	except Exception:
		default_max_n_val = 0
	max_sources = int(os.getenv("DATASET_LIMIT_SOURCES", "0") or "0")  # 0 means no limit

	seen: List[str] = []
	if train_new_only and state_path:
		try:
			st = _read_json(Path(state_path))
			seen = list(st.get("datasets_seen") or [])
		except Exception:
			seen = []

	count_sources = 0
	for src in sources:
		name = str(src.get("name") or "").strip()
		if not name:
			continue
		if train_new_only and name in seen:
			continue
		if full_datasets:
			max_n = None
		else:
			max_n = src.get("max_n")
			if max_n is None and default_max_n_val > 0:
				max_n = default_max_n_val
		text_key = str(src.get("text_key") or "text").strip()
		subset = src.get("subset")
		split = str(src.get("split") or "train").strip()
		rel = str(src.get("path") or "").strip()
		if rel:
			fp = Path(rel)
			if not fp.is_absolute():
				fp = (local_dir / rel).resolve()
		else:
			suffix = f"_{split}" if split else ""
			fp = (local_dir / f"{name.replace('/', '_')}{suffix}.jsonl").resolve()
		out_texts: List[str] = _load_local_jsonl(fp, text_key=text_key, max_n=max_n)
		if not out_texts:
			out_texts = _load_hf(
				name,
				subset,
				split,
				max_n,
				cache_dir,
				streaming=streaming,
				prompt_key=(src.get("prompt_key") if isinstance(src.get("prompt_key"), str) else None),
				code_key=(src.get("code_key") if isinstance(src.get("code_key"), str) else None),
				compose=(src.get("compose") if isinstance(src.get("compose"), list) else None),
				join_with=str(src.get("join_with") or "\n"),
			)
			if out_texts:
				try:
					with open(fp, "w", encoding="utf-8") as fh:
						for t in out_texts:
							fh.write(json.dumps({"text": t}) + "\n")
				except Exception:
					pass
		if out_texts:
			yield out_texts, (f"{name}:{split}" if split else name)
			count_sources += 1
			if max_sources and count_sources >= max_sources:
				break

