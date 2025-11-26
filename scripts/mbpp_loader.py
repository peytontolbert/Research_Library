from __future__ import annotations

from typing import List, Tuple


def load_mbpp_texts(max_n: int = 128, split: str = "train") -> Tuple[List[str], int]:
	"""
	Load MBPP via Hugging Face datasets and return a list of training texts.
	Each text is prompt + solution combined to form a code-aware LM sample.
	Returns (texts, total_available).
	"""
	try:
		from datasets import load_dataset  # type: ignore
	except Exception:
		return [], 0
	# Try common configs in order of availability
	ds = None
	for name in ("mbpp",):
		for subset in ("sanitized", None):
			try:
				if subset is None:
					ds = load_dataset(name, split=split)
				else:
					ds = load_dataset(name, subset, split=split)
				break
			except Exception:
				continue
		if ds is not None:
			break
	if ds is None:
		return [], 0
	texts: List[str] = []
	total = len(ds)
	for i in range(total):
		row = ds[i]
		# MBPP variants have fields like: "text" or "prompt"/"question", "code" (solution), "test_list"
		prompt = str(row.get("prompt") or row.get("question") or row.get("text") or "").strip()
		code = str(row.get("code") or row.get("solution") or "").strip()
		if not prompt and not code:
			continue
		# Simple concatenation; training LM on both description and solution
		combined = (prompt + "\n" + code).strip()
		if combined:
			texts.append(combined)
		if len(texts) >= int(max_n):
			break
	return texts, total


def load_mbpp_texts_all_splits(max_n: int = 1_000_000) -> Tuple[List[str], int]:
	"""
	Load MBPP across all available splits (e.g., train/validation/test) and concatenate.
	Returns (texts, total_available_across_splits).
	"""
	try:
		from datasets import load_dataset  # type: ignore
	except Exception:
		return [], 0
	ds_dict = None
	for name in ("mbpp",):
		for subset in ("sanitized", None):
			try:
				if subset is None:
					ds_dict = load_dataset(name)  # DatasetDict
				else:
					ds_dict = load_dataset(name, subset)  # DatasetDict
				break
			except Exception:
				continue
		if ds_dict is not None:
			break
	if ds_dict is None:
		return [], 0
	texts: List[str] = []
	total = 0
	# Iterate splits in a stable order
	for split_name in ("train", "validation", "test"):
		if split_name not in ds_dict:
			continue
		ds = ds_dict[split_name]
		sz = len(ds)
		total += sz
		for i in range(sz):
			row = ds[i]
			prompt = str(row.get("prompt") or row.get("question") or row.get("text") or "").strip()
			code = str(row.get("code") or row.get("solution") or "").strip()
			if not prompt and not code:
				continue
			combined = (prompt + "\n" + code).strip()
			if combined:
				texts.append(combined)
			if len(texts) >= int(max_n):
				return texts, total
	return texts, total


