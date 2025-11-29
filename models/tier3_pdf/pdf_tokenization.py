"""
Model 10: PDF Tokenization / Structuring (P0).
Turns PDF files into structured tokens: text, equations, tables, figures, references.
Uses layout-aware extraction when available; can optionally use a vision-language
OCR model (e.g., Qwen/Qwen3-VL-2B-Instruct) when configured; falls back to line
heuristics. HF generation remains available via the wrapper for downstream fine-
tuning, but the tokenize() method provides the structured output expected by PLAN.md.
"""

from typing import Any, Dict, List, Optional, Tuple

from models.shared.modeling import GenerativeModel
from models.shared.pdf_utils import extract_pdf_text, extract_pdf_structure
from pathlib import Path

try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:
    convert_from_path = None

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None

try:
    from transformers import AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except Exception:
    AutoProcessor = None  # type: ignore
    Qwen3VLForConditionalGeneration = None  # type: ignore

try:
    import torch
except Exception:
    torch = None


class PDFTokenizationModel(GenerativeModel):
    """Hybrid P0: structured tokenizer with HF-backed generative path."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    @staticmethod
    def _classify_line(line: str, in_references: bool) -> str:
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            return "blank"

        if in_references:
            return "reference"

        if lower.startswith("references") or lower.startswith("bibliography"):
            return "section_header"

        if "|" in line or "\t" in line or lower.startswith("table "):
            return "table"

        if lower.startswith("figure ") or lower.startswith("fig."):
            return "figure"

        math_markers = ("=", "\\sum", "\\int", "\\frac", "\\alpha", "\\beta", "\\gamma")
        if any(marker in line for marker in math_markers):
            return "equation"

        return "text"

    _qwen_cache: Dict[str, Tuple[Any, Any]] = {}

    def _ocr_with_qwen_pipeline(self, pdf_path: str, model_name: Optional[str]) -> List[Dict[str, Any]]:
        """Optional OCR using generic image-to-text pipeline."""
        if model_name is None or pipeline is None or convert_from_path is None:
            return []
        try:
            vlm = pipeline("image-to-text", model=model_name, trust_remote_code=True)
        except Exception:
            return []
        tokens: List[Dict[str, Any]] = []
        try:
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=3)
        except Exception:
            return []
        for page_idx, img in enumerate(images, start=1):
            try:
                result = vlm(img)
            except Exception:
                continue
            text = ""
            if isinstance(result, list) and result and isinstance(result[0], dict):
                text = result[0].get("generated_text") or result[0].get("text") or ""
            elif isinstance(result, str):
                text = result
            if text:
                tokens.append(
                    {
                        "type": "text",
                        "text": text.strip(),
                        "page": page_idx,
                        "source": "qwen_vlm",
                    }
                )
        return tokens

    def _load_qwen3(self, model_name: str):
        if model_name in self._qwen_cache:
            return self._qwen_cache[model_name]
        if AutoProcessor is None or Qwen3VLForConditionalGeneration is None or torch is None:
            return None, None
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, dtype="auto", device_map="auto", trust_remote_code=True
            )
            self._qwen_cache[model_name] = (processor, model)
            return processor, model
        except Exception:
            return None, None

    def _ocr_with_qwen3(self, pdf_path: str, model_name: Optional[str]) -> List[Dict[str, Any]]:
        """OCR using Qwen3-VL style models with image+text prompting."""
        if model_name is None or convert_from_path is None:
            return []
        processor, model = self._load_qwen3(model_name)
        if processor is None or model is None:
            return []
        try:
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=3)
        except Exception:
            return []
        tokens: List[Dict[str, Any]] = []
        for page_idx, img in enumerate(images, start=1):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Extract all readable text from this page as plain text."},
                    ],
                }
            ]
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                # Force GPU when available; otherwise use model.device
                target_device = model.device
                if torch.cuda.is_available():
                    target_device = torch.device("cuda")
                    model.to(target_device)
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                gen = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=0.8,
                    top_k=20,
                    temperature=0.7,
                    repetition_penalty=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
                gen_trim = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], gen)]
                text_list = processor.batch_decode(
                    gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                text = text_list[0] if text_list else ""
            except Exception:
                text = ""
            if text:
                tokens.append(
                    {
                        "type": "text",
                        "text": text.strip(),
                        "page": page_idx,
                        "source": "qwen3_vlm",
                    }
                )
        return tokens

    def tokenize(
        self,
        pdf_path: str,
        vlm_model: Optional[str] = None,
        max_pages: int = 3,
        use_ocr: bool = True,
        use_clip: bool = True,
    ) -> List[Dict[str, Any]]:
        """Convert a PDF (or text file) into a list of structured tokens."""
        if not isinstance(pdf_path, str):
            raise TypeError(f"pdf_path must be a string path, got {type(pdf_path)!r}")

        # Optional OCR/VLM path for better text extraction.
        ocr_tokens: List[Dict[str, Any]] = []
        if vlm_model:
            # Prefer Qwen3 flow for Qwen models; fallback to generic pipeline.
            if "qwen" in vlm_model.lower():
                ocr_tokens = self._ocr_with_qwen3(pdf_path, vlm_model)
            if not ocr_tokens:
                ocr_tokens = self._ocr_with_qwen_pipeline(pdf_path, vlm_model)
        if ocr_tokens:
            return ocr_tokens

        # Prefer layout-aware tokens when available.
        struct_tokens = extract_pdf_structure(pdf_path, max_pages=max_pages, max_items=400, use_ocr=use_ocr, use_clip=use_clip)
        if struct_tokens:
            return struct_tokens

        raw = extract_pdf_text(pdf_path, max_chars=50000)
        tokens: List[Dict[str, Any]] = []
        in_references = False
        for idx, raw_line in enumerate(raw.splitlines(), start=1):
            kind = self._classify_line(raw_line, in_references)

            if kind == "section_header":
                in_references = True
                tokens.append({"type": "text", "text": raw_line.rstrip("\n"), "line_no": idx})
                continue

            if kind == "blank":
                continue

            if in_references and kind == "text":
                kind = "reference"

            tokens.append({"type": kind, "text": raw_line.rstrip("\n"), "line_no": idx})

        return tokens
