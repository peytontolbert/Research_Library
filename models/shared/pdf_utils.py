"""
Minimal PDF text extraction helpers.

Order of preference:
- PyMuPDF (fitz) if available, for better layout-aware extraction and layout blocks.
- PyPDF2 as a fallback.
- Permissive binary decode as a last resort.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional

try:
    import PyPDF2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PyPDF2 = None
try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CLIPModel = None
    CLIPProcessor = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None

_CLIP_CACHE = None  # (processor, model)


def _get_clip():
    """Lazy-load CLIP model/processor if available locally."""
    global _CLIP_CACHE
    if _CLIP_CACHE or CLIPModel is None or CLIPProcessor is None:
        return _CLIP_CACHE
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        if torch is not None and torch.cuda.is_available():
            model.to("cuda")
        _CLIP_CACHE = (processor, model)
    except Exception:
        _CLIP_CACHE = None
    return _CLIP_CACHE


def extract_pdf_text(path: str, *, max_chars: int = 5000) -> str:
    """Extract text from a PDF, returning at most `max_chars`."""
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        try:
            return p.read_text(encoding="utf-8")[:max_chars]
        except Exception:
            try:
                return p.read_bytes().decode("latin-1", errors="ignore")[:max_chars]
            except Exception:
                return ""

    # Preferred: poppler pdftotext if available.
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", "-q", str(p), "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.stdout:
            return proc.stdout[:max_chars]
    except Exception:
        pass

    if fitz is not None:
        try:
            doc = fitz.open(p)
            parts = []
            for page in doc:
                if len("".join(parts)) >= max_chars:
                    break
                parts.append(page.get_text("text"))
            doc.close()
            if parts:
                return "".join(parts)[:max_chars]
        except Exception:
            pass

    if PyPDF2 is None:
        try:
            return p.read_bytes().decode("latin-1", errors="ignore")[:max_chars]
        except Exception:
            return f"PDF_PATH::{path}"

    try:
        text_parts = []
        with p.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                if len("".join(text_parts)) >= max_chars:
                    break
                try:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                except Exception:
                    continue
        return "".join(text_parts)[:max_chars]
    except Exception:
        return f"PDF_PATH::{path}"


def extract_pdf_structure(path: str, *, max_pages: int = 3, max_items: int = 200, use_ocr: bool = True, use_clip: bool = True) -> list:
    """
    Extract coarse layout-aware tokens from a PDF using fitz if available.

    Returns a list of dicts with type, text, bbox, and page.
    Types: text, heading, table-ish, figure (image placeholder).
    """
    tokens: list = []
    p = Path(path)
    if fitz is None or p.suffix.lower() != ".pdf":
        # Fallback: treat as plain text lines
        text = extract_pdf_text(path, max_chars=20000)
        for idx, line in enumerate(text.splitlines(), start=1):
            tokens.append({"type": "text", "text": line, "page": 1, "bbox": None, "line_no": idx})
        return tokens[:max_items]

    try:
        doc = fitz.open(p)
        for page_index, page in enumerate(doc[:max_pages], start=1):
            blocks = page.get_text("blocks")
            for b in blocks:
                bbox = b[:4]
                text = (b[4] or "").strip()
                if not text:
                    continue
                lower = text.lower()
                kind = "text"
                if len(text.split()) <= 10 and any(lower.startswith(h) for h in ("abstract", "introduction", "method", "results", "conclusion", "references")):
                    kind = "heading"
                elif "|" in text or "\t" in text:
                    kind = "table"
                tokens.append({"type": kind, "text": text, "page": page_index, "bbox": bbox})
                if len(tokens) >= max_items:
                    doc.close()
                    return tokens
            # Add image placeholders
            try:
                images = page.get_images(full=True)
                for _img in images:
                    ocr_text = ""
                    embedding = None
                    if use_ocr and pytesseract is not None:
                        try:
                            pix = fitz.Pixmap(doc, _img[0])
                            if pix.alpha:  # remove alpha
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            ocr_text = pytesseract.image_to_string(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
                        except Exception:
                            ocr_text = ""
                    # Optional CLIP embedding
                    if use_clip:
                        clip_pair = _get_clip()
                        if clip_pair is not None and torch is not None:
                            try:
                                processor, model = clip_pair
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                inputs = processor(images=img, return_tensors="pt")
                                if torch.cuda.is_available():
                                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                                with torch.no_grad():
                                    feats = model.get_image_features(**inputs)
                                feats = model.visual_projection(feats)
                                feats = torch.nn.functional.normalize(feats, dim=-1)
                                embedding = feats[0].cpu().tolist()
                            except Exception:
                                embedding = None
                    tokens.append({"type": "figure", "text": ocr_text or "[IMAGE]", "page": page_index, "bbox": None, "embedding": embedding})
                    if len(tokens) >= max_items:
                        doc.close()
                        return tokens
            except Exception:
                pass
        doc.close()
    except Exception:
        pass
    return tokens[:max_items]
