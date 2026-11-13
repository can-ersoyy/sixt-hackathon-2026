"""Stage 1 Adaptive Modality Router, Document Chunker, and Schema Chunker.

Implements:
1. PipelineConfig: Configuration dataclass for routing, chunking, and model thresholds.
2. RouteDecision: Routing decision object (modality, model, images, companion text, chunks).
3. determine_route(): Selects VLM vs. Text model based on document type, length, and page count.
4. Document Chunking: Splits long documents (>max_doc_chars) along section boundaries with overlap.
5. merge_chunk_predictions(): Merges predictions across chunks (highest confidence non-null).
6. Schema Chunking: Splits dense schemas (>schema_chunk_size) and merges sub-predictions.
"""
from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

try:
    from stage1_pipeline.parser import ParsedDocument, render_pdf_pages_base64
except ImportError:
    from parser import ParsedDocument, render_pdf_pages_base64

try:
    from stage1_pipeline.logging_utils import get_logger, log_route_decision
except ImportError:
    try:
        from logging_utils import get_logger, log_route_decision
    except ImportError:
        get_logger = logging.getLogger
        log_route_decision = None

logger = get_logger("stage1_pipeline.router")


@dataclass
class PipelineConfig:
    """Configuration for routing, chunking, and extraction."""
    route_mode: str = "auto"  # "auto" | "vlm" | "text"
    vlm_max_pages: int = 5
    max_doc_chars: int = 180_000
    chunk_overlap_chars: int = 5_000
    schema_chunk_size: int = 20
    concurrency: int = 10
    model_text: str = "gpt-4o-mini"
    model_vlm: str = "gpt-4o-mini"
    model_pass2: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:
        if "extra" in self.__dict__ and item in self.__dict__["extra"]:
            return self.__dict__["extra"][item]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        return self.extra.get(key, default)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PipelineConfig:
        known = {
            "route_mode",
            "vlm_max_pages",
            "max_doc_chars",
            "chunk_overlap_chars",
            "schema_chunk_size",
            "concurrency",
            "model_text",
            "model_vlm",
            "model_pass2",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class RouteDecision:
    """Decision produced by determine_route."""
    modality: str  # "vlm" | "text"
    model: str  # Model identifier to invoke
    images_b64: List[str] = field(default_factory=list)
    companion_text: Optional[str] = None
    doc_chunks: List[str] = field(default_factory=list)
    reason: str = ""
    primary_path: Optional[Path] = None


def resolve_file_path(path_str: str | Path) -> Optional[Path]:
    """Resolves a document path against DATA_DIR, ROOT_DIR, or absolute path."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for base in (DATA_DIR, ROOT_DIR, Path(".")):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return None


def determine_route(
    rec: dict,
    parsed: ParsedDocument,
    config: Optional[PipelineConfig] = None,
) -> RouteDecision:
    """Determines whether a document should be processed via VLM or Text model.

    Routing Rules:
    - If config.route_mode == 'vlm', force VLM.
    - If config.route_mode == 'text', force Text.
    - If config.route_mode == 'auto':
      - VLM: doc_type in ('scanned_pdf', 'image') OR (doc_type == 'digital_pdf' and page_count <= vlm_max_pages).
      - Text: doc_type == 'cuad' OR page_count > vlm_max_pages.

    When VLM is chosen:
    - Renders page images via render_pdf_pages_base64 (or reads raw image file).
    - If doc_type == 'digital_pdf', attaches parsed.layout_text as companion text.

    When Text is chosen:
    - Splits parsed.layout_text into chunks if > config.max_doc_chars.
    """
    if config is None:
        config = PipelineConfig()
    elif isinstance(config, dict):
        config = PipelineConfig.from_dict(config)

    # Resolve primary file path if present
    pages = rec.get("pages", [])
    primary_path: Optional[Path] = None
    if pages:
        primary_path = resolve_file_path(pages[0])

    mode = config.route_mode.lower()
    selected_modality: str
    reason: str

    if mode == "vlm":
        selected_modality = "vlm"
        reason = "Manual override: route_mode='vlm'"
    elif mode == "text":
        selected_modality = "text"
        reason = "Manual override: route_mode='text'"
    else:  # "auto"
        if parsed.doc_type in ("scanned_pdf", "image"):
            selected_modality = "vlm"
            reason = f"Scanned/image document ({parsed.doc_type}) -> VLM"
        elif parsed.doc_type == "digital_pdf" and parsed.page_count <= config.vlm_max_pages:
            selected_modality = "vlm"
            reason = f"Short digital PDF ({parsed.page_count} pages <= {config.vlm_max_pages}) -> VLM"
        elif parsed.doc_type == "cuad":
            selected_modality = "text"
            reason = "CUAD legal contract -> Text Model (annotated node tree)"
        elif parsed.page_count > config.vlm_max_pages:
            selected_modality = "text"
            reason = f"Long document ({parsed.page_count} pages > {config.vlm_max_pages}) -> Text Model"
        else:
            selected_modality = "text"
            reason = f"Default fallback for {parsed.doc_type} -> Text Model"

    if selected_modality == "vlm":
        model = config.model_vlm
        images_b64: List[str] = []
        companion_text = parsed.layout_text if parsed.doc_type == "digital_pdf" else None

        if primary_path and primary_path.exists():
            ext = primary_path.suffix.lower()
            if ext == ".pdf":
                try:
                    images_b64 = render_pdf_pages_base64(
                        primary_path, max_pages=config.vlm_max_pages
                    )
                except Exception as e:
                    logger.warning("Failed to render PDF pages for %s: %s", primary_path, e)
            elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"):
                # Load images for each listed page
                for p_str in pages[: config.vlm_max_pages]:
                    p_path = resolve_file_path(p_str)
                    if p_path and p_path.exists():
                        try:
                            with open(p_path, "rb") as f:
                                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
                        except Exception as e:
                            logger.warning("Failed to read image %s: %s", p_path, e)

        decision = RouteDecision(
            modality="vlm",
            model=model,
            images_b64=images_b64,
            companion_text=companion_text,
            doc_chunks=[],
            reason=reason,
            primary_path=primary_path,
        )

    else:  # Text modality
        model = config.model_text
        doc_chunks = chunk_document_text(
            parsed.layout_text,
            max_chars=config.max_doc_chars,
            overlap_chars=config.chunk_overlap_chars,
        )

        decision = RouteDecision(
            modality="text",
            model=model,
            images_b64=[],
            companion_text=None,
            doc_chunks=doc_chunks,
            reason=reason,
            primary_path=primary_path,
        )

    if log_route_decision is not None:
        log_route_decision(
            doc_id=rec.get("doc_id", getattr(parsed, "doc_id", "unknown")),
            modality=decision.modality,
            model=decision.model,
            chunks=len(decision.doc_chunks) if decision.doc_chunks else 1,
            images=len(decision.images_b64),
            logger=logger,
        )

    return decision


def chunk_document_text(
    text: str,
    max_chars: int = 180_000,
    overlap_chars: int = 5_000,
) -> List[str]:
    """Splits document text along section boundaries with overlap.

    If len(text) <= max_chars, returns [text].
    Otherwise, splits along [sec-X] headings or double newlines,
    ensuring each chunk is within max_chars and consecutive chunks
    overlap by approximately overlap_chars.
    """
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    # Split into units along section headings
    section_regex = re.compile(r"(?:\n{1,2})(?=\[sec-\d+)")
    parts = [p.strip() for p in section_regex.split(text) if p.strip()]

    # If no section headings found, split by double newlines
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Further sub-divide any unit that individually exceeds max_chars
    units: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            units.append(part)
        else:
            sub_parts = [sp.strip() for sp in part.split("\n\n") if sp.strip()]
            if len(sub_parts) > 1:
                for sp in sub_parts:
                    if len(sp) <= max_chars:
                        units.append(sp)
                    else:
                        lines = [l for l in sp.split("\n") if l.strip()]
                        units.extend(lines if lines else [sp])
            else:
                lines = [l for l in part.split("\n") if l.strip()]
                if len(lines) > 1:
                    units.extend(lines)
                else:
                    # Hard slice if a single line exceeds max_chars
                    step = max(1000, max_chars - overlap_chars)
                    for i in range(0, len(part), step):
                        units.append(part[i : i + max_chars])

    chunks: List[str] = []
    current_units: List[str] = []
    current_len = 0
    separator = "\n\n"

    i = 0
    while i < len(units):
        unit = units[i]
        unit_len = len(unit) + (len(separator) if current_units else 0)

        if current_units and (current_len + unit_len > max_chars):
            # Finalize current chunk
            chunks.append(separator.join(current_units))

            # Build overlap from the end of current_units
            overlap_units: List[str] = []
            overlap_len = 0
            for u in reversed(current_units):
                u_len = len(u) + (len(separator) if overlap_units else 0)
                if overlap_len + u_len <= overlap_chars:
                    overlap_units.insert(0, u)
                    overlap_len += u_len
                else:
                    if not overlap_units and len(u) < max_chars:
                        overlap_units.insert(0, u)
                    break

            # Avoid stuck loops if overlap covers all units
            if len(overlap_units) >= len(current_units):
                overlap_units = []

            current_units = list(overlap_units)
            current_len = (
                sum(len(u) for u in current_units)
                + (len(separator) * (len(current_units) - 1) if len(current_units) > 1 else 0)
            )

            if unit in current_units:
                i += 1
                continue
        else:
            current_units.append(unit)
            current_len += unit_len
            i += 1

    if current_units:
        chunks.append(separator.join(current_units))

    return chunks if chunks else [text]


def is_non_null_value(val: Any) -> bool:
    """Determines if an extracted field value is non-null and meaningful."""
    if val is None:
        return False
    if isinstance(val, bool):
        return True  # Boolean False is a valid prediction (e.g. unchecked checkbox)
    if isinstance(val, str):
        s = val.strip()
        sl = s.lower()
        if not sl or sl in ("null", "none", "n/a", "na", "not applicable", "not specified") or sl.startswith("none ("):
            return False
        return True
    return True


def merge_chunk_predictions(chunk_predictions: List[Dict[str, dict]]) -> Dict[str, dict]:
    """Merges field predictions across multiple document chunks.

    For each field:
    - Chooses the non-null prediction with the highest confidence.
    - If all chunks returned null (or empty), emits value=None with confidence=0.0.
    """
    if not chunk_predictions:
        return {}

    all_fields: Set[str] = set()
    for cp in chunk_predictions:
        if isinstance(cp, dict):
            target = cp.get("fields") if isinstance(cp.get("fields"), dict) else cp
            all_fields.update(target.keys())

    merged: Dict[str, dict] = {}
    for field_name in all_fields:
        best_cand: Optional[dict] = None
        best_conf: float = -1.0

        for cp in chunk_predictions:
            if not isinstance(cp, dict):
                continue
            target = cp.get("fields") if isinstance(cp.get("fields"), dict) else cp
            cand = target.get(field_name)
            if not isinstance(cand, dict):
                continue

            val = cand.get("value")
            if is_non_null_value(val):
                conf = float(cand.get("confidence", 0.0) or 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_cand = cand

        if best_cand is not None:
            merged[field_name] = dict(best_cand)
        else:
            merged[field_name] = {
                "value": None,
                "confidence": 0.0,
                "quote": None,
                "evidence_span": None,
            }

    return merged


def split_schema(schema: dict, chunk_size: int = 20) -> List[dict]:
    """Splits schema['properties'] into sub-schemas of size <= chunk_size,
    preserving schema metadata ($defs, title, type, $schema, etc.).
    """
    if not isinstance(schema, dict):
        return [schema]

    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or len(properties) <= chunk_size:
        return [dict(schema)]

    prop_items = list(properties.items())
    sub_schemas: List[dict] = []
    total_chunks = (len(prop_items) + chunk_size - 1) // chunk_size

    for i in range(0, len(prop_items), chunk_size):
        chunk_props = dict(prop_items[i : i + chunk_size])
        sub_schema = dict(schema)
        sub_schema["properties"] = chunk_props

        if "required" in schema and isinstance(schema["required"], list):
            sub_schema["required"] = [r for r in schema["required"] if r in chunk_props]

        # Add chunk tracking metadata
        metadata = dict(sub_schema.get("metadata", {})) if isinstance(sub_schema.get("metadata"), dict) else {}
        metadata["chunk_index"] = len(sub_schemas)
        metadata["total_chunks"] = total_chunks
        sub_schema["metadata"] = metadata

        sub_schemas.append(sub_schema)

    return sub_schemas


def merge_schema_predictions(sub_predictions: List[Dict[str, dict]]) -> Dict[str, dict]:
    """Combines field predictions from all schema chunks into a unified fields dictionary."""
    merged: Dict[str, dict] = {}
    if not sub_predictions:
        return merged

    for sub_pred in sub_predictions:
        if not isinstance(sub_pred, dict):
            continue
        target = sub_pred.get("fields") if isinstance(sub_pred.get("fields"), dict) else sub_pred
        for k, v in target.items():
            if k == "doc_id":
                continue
            merged[k] = v

    return merged
