"""Stage 1 Document Identification & Layout-Aware Parsing.

Produces clean, layout-aware markdown and a hierarchical Node Tree
with canonical path annotations ([sec-1/p-2/s-5], [sec-2/table-0/row-3], etc.).
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root and package directory are in sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from stage1_pipeline.node_tree import (
        Node,
        SectionNode,
        ParagraphNode,
        SentenceNode,
        TableNode,
        TableRowNode,
        build_tree_from_markdown,
    )
except ImportError:
    from node_tree import (
        Node,
        SectionNode,
        ParagraphNode,
        SentenceNode,
        TableNode,
        TableRowNode,
        build_tree_from_markdown,
    )

try:
    from stage1_pipeline.logging_utils import get_logger, log_document_start
except ImportError:
    try:
        from logging_utils import get_logger, log_document_start
    except ImportError:
        get_logger = logging.getLogger
        log_document_start = None

logger = get_logger("stage1_pipeline.parser")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

_CUAD_CACHE: Optional[Dict[str, str]] = None
_DOCLING_CONVERTER: Any = None
_DOCLING_TRIED: bool = False


@dataclass
class ParsedDocument:
    """Modular representation of a parsed document."""
    doc_id: str
    doc_type: str  # 'cuad', 'digital_pdf', 'scanned_pdf', 'image'
    source_dataset: str
    layout_text: str  # Markdown text with embedded canonical path tags
    raw_text: str  # Clean raw text without annotations
    page_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    nodes: Dict[str, str] = field(default_factory=dict)  # canonical path -> text
    node_tree: Optional[Node] = None
    tables: List[Dict[str, Any]] = field(default_factory=list)

    def get_node_text(self, path: str) -> Optional[str]:
        """Retrieve text of a node using its canonical path tag e.g. [sec-1/p-2/s-5]."""
        return self.nodes.get(path) or self.nodes.get(path.strip("[]"))

    def get_node_with_neighbors(self, path: str) -> str:
        """Retrieve text of a node along with its immediate previous and next siblings.
        
        Args:
            path: Canonical path tag, e.g. '[sec-1/p-2/s-5]' or 'sec-1/p-2/s-5'.
            
        Returns:
            The combined text of previous sibling (if any), current node, and
            next sibling (if any) in its parent's children list, joined with spaces/newlines.
            If the node has no parent, no siblings, or is only present in self.nodes,
            falls back gracefully to self.get_node_text(path) or ''.
        """
        if not path:
            return ""

        fallback_text = self.get_node_text(path) or ""

        if self.node_tree is None:
            return fallback_text

        # Find target node via node_tree method or fallback traversal
        if hasattr(self.node_tree, "find_node_by_path"):
            target_node = self.node_tree.find_node_by_path(path)
        else:
            target_node = self._find_node_by_path(self.node_tree, path.strip("[]"))

        if not target_node or not target_node.parent:
            return fallback_text

        siblings = target_node.parent.children
        if len(siblings) <= 1:
            return fallback_text

        try:
            idx = siblings.index(target_node)
        except ValueError:
            return fallback_text

        prev_node = siblings[idx - 1] if idx > 0 else None
        next_node = siblings[idx + 1] if idx + 1 < len(siblings) else None

        if prev_node is None and next_node is None:
            return fallback_text

        def _get_text(n: Node) -> str:
            # Check self.nodes first for consistency with get_node_text
            txt = self.get_node_text(n.canonical_path)
            if txt and txt.strip():
                return txt.strip()
            if n.text and n.text.strip():
                return n.text.strip()
            return n.to_raw_text().strip()

        ordered_nodes = []
        if prev_node is not None:
            ordered_nodes.append(prev_node)
        ordered_nodes.append(target_node)
        if next_node is not None:
            ordered_nodes.append(next_node)

        texts = []
        for n in ordered_nodes:
            t = _get_text(n)
            if t:
                texts.append(t)

        if not texts:
            return fallback_text

        all_sentences = all(getattr(n, "node_type", "") == "sentence" for n in ordered_nodes)
        any_newline = any("\n" in t for t in texts)
        if all_sentences and not any_newline:
            return " ".join(texts)
        elif any_newline or any(getattr(n, "node_type", "") in ("table_row", "table", "section", "paragraph") for n in ordered_nodes):
            return "\n".join(texts)
        else:
            return " ".join(texts)

    def _find_node_by_path(self, node: Node, target_clean: str) -> Optional[Node]:
        if node.canonical_path.strip("[]") == target_clean:
            return node
        for child in node.children:
            res = self._find_node_by_path(child, target_clean)
            if res is not None:
                return res
        return None


def load_cuad_corpus() -> Dict[str, str]:
    """Loads full CUAD contract contexts from data/raw/cuad/CUADv1.json.
    
    COMPLIANCE WITH AGENTS.md:
    ZERO TOLERANCE FOR CHEATING.
    Only reads 'context' from d['paragraphs']. Never inspects 'qas' or 'answers'.
    """
    global _CUAD_CACHE
    if _CUAD_CACHE is None:
        cuad_path = DATA_DIR / "raw/cuad/CUADv1.json"
        if not cuad_path.exists():
            raise FileNotFoundError(f"CUAD file not found at {cuad_path}")
        raw = json.loads(cuad_path.read_text(encoding="utf-8"))["data"]
        # Strictly extract ONLY paragraph context. Do not touch qas!
        _CUAD_CACHE = {
            d["title"]: "\n\n".join(p["context"] for p in d["paragraphs"])
            for d in raw
        }
    return _CUAD_CACHE


def get_docling_converter():
    """Lazily initializes and caches the Docling DocumentConverter."""
    global _DOCLING_CONVERTER, _DOCLING_TRIED
    if not _DOCLING_TRIED:
        _DOCLING_TRIED = True
        try:
            from docling.document_converter import DocumentConverter
            _DOCLING_CONVERTER = DocumentConverter()
            logger.info("Docling DocumentConverter initialized successfully.")
        except Exception as e:
            logger.warning("Docling initialization failed, falling back to pypdf/pdftotext: %s", e)
            _DOCLING_CONVERTER = None
    return _DOCLING_CONVERTER


def identify_document_type(rec: dict) -> Tuple[str, Optional[Path], int]:
    """Identifies document category: 'cuad', 'digital_pdf', 'scanned_pdf', or 'image'.
    
    Returns:
        (doc_type, primary_file_path, page_count)
    """
    source = rec.get("source_dataset", "").lower()
    doc_id = rec.get("doc_id", "")

    if source == "cuad" or doc_id.startswith("cuad:"):
        return "cuad", None, 1

    pages = rec.get("pages", [])
    if not pages:
        return "unknown", None, 0

    page_path = DATA_DIR / pages[0]
    ext = page_path.suffix.lower()

    if ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"):
        return "image", page_path, len(pages)

    if ext == ".pdf":
        page_count = 1
        digital_chars = 0
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(page_path))
            page_count = len(reader.pages)
            for p in reader.pages:
                txt = p.extract_text() or ""
                digital_chars += len(txt.strip())
        except Exception:
            pass

        if digital_chars > 50:
            return "digital_pdf", page_path, page_count
        else:
            return "scanned_pdf", page_path, page_count

    return "unknown", page_path, len(pages)


def _parse_pdf_fallback(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Graceful fallback parser for PDFs using pdftotext or pypdf."""
    # Try pdftotext with layout preservation
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout, []

    # Fallback to pypdf layout extraction
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                t = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
            except Exception:
                t = page.extract_text() or ""
            if t.strip():
                pages.append(f"## Page {i}\n\n{t}")
        return "\n\n".join(pages), []
    except Exception as e:
        logger.error("PDF fallback failed for %s: %s", path, e)
        return "", []


def _parse_with_docling(path: Path) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Parses a document with Docling, exporting layout markdown and table structures."""
    converter = get_docling_converter()
    if converter is None:
        return None, []

    try:
        result = converter.convert(str(path))
        doc = result.document
        md = doc.export_to_markdown()

        tables_data = []
        if hasattr(doc, "tables"):
            for i, tbl in enumerate(doc.tables):
                try:
                    df = tbl.export_to_dataframe(doc=doc)
                    tables_data.append({
                        "table_id": f"table-{i}",
                        "shape": list(df.shape),
                        "columns": [str(c) for c in df.columns],
                        "rows": df.values.tolist(),
                    })
                except Exception:
                    pass

        return md, tables_data
    except Exception as e:
        logger.warning("Docling conversion error on %s: %s", path, e)
        return None, []


def parse_document(rec: dict) -> ParsedDocument:
    """Modular parsing entrypoint for any unified document record.
    
    Returns a ParsedDocument containing:
    - doc_id and doc_type ('cuad', 'digital_pdf', 'scanned_pdf', 'image')
    - layout_text: clean markdown with canonical path tags ([sec-1/p-2/s-5], [sec-2/table-0/row-3])
    - raw_text: clean verbatim document text for grounding checks
    - node_tree: hierarchical tree of SectionNode, ParagraphNode, SentenceNode, TableNode
    - nodes: index mapping canonical path strings to their node text
    - metadata & extracted tables
    """
    doc_id = rec["doc_id"]
    doc_type, primary_path, page_count = identify_document_type(rec)
    metadata: Dict[str, Any] = {
        "doc_type": doc_type,
        "page_count": page_count,
        "parser": None,
        "ocr_used": False,
    }
    tables_data: List[Dict[str, Any]] = []

    # 1. CUAD Legal Contract Processing
    if doc_type == "cuad":
        corpus = load_cuad_corpus()
        contract_title = doc_id.split(":", 1)[1] if ":" in doc_id else doc_id
        if contract_title not in corpus:
            raise KeyError(f"Contract title '{contract_title}' not found in CUAD corpus")
        raw_content = corpus[contract_title]  # Full text, NO 60k character truncation!
        metadata["parser"] = "cuad_corpus"
        metadata["char_count"] = len(raw_content)

        if log_document_start is not None:
            log_document_start(
                doc_id=doc_id,
                doc_type=doc_type,
                pages=1,
                chars=len(raw_content),
                logger=logger,
            )

        tree = build_tree_from_markdown(raw_content)
        annotated = tree.to_annotated_text()
        index = tree.build_index()

        return ParsedDocument(
            doc_id=doc_id,
            doc_type=doc_type,
            source_dataset="cuad",
            layout_text=annotated,
            raw_text=raw_content,
            page_count=1,
            metadata=metadata,
            nodes=index,
            node_tree=tree,
            tables=[],
        )

    # 2. ExtractBench PDF / Image Processing
    if primary_path is None or not primary_path.exists():
        raise FileNotFoundError(f"Document file missing: {primary_path}")

    md_text = None
    # For digital PDFs, use fast layout extraction (0.05s) first to avoid CPU bottleneck
    if doc_type == "digital_pdf":
        fast_text, tables = _parse_pdf_fallback(primary_path)
        if fast_text and len(fast_text.strip()) > 50:
            md_text = fast_text
            metadata["parser"] = "fast_pdf"
            metadata["ocr_used"] = False
            metadata["has_tables"] = False

    # For scanned PDFs or when fast extraction is insufficient, invoke Docling
    if not md_text:
        md_text, tables = _parse_with_docling(primary_path)
        if md_text and md_text.strip():
            metadata["parser"] = "docling"
            metadata["ocr_used"] = (doc_type == "scanned_pdf")
            metadata["has_tables"] = len(tables) > 0
            tables_data = tables
        else:
            md_text, tables = _parse_pdf_fallback(primary_path)
            metadata["parser"] = "fallback_pdf"
            metadata["ocr_used"] = False
            metadata["has_tables"] = False

    raw_text = md_text or ""
    metadata["char_count"] = len(raw_text)

    if log_document_start is not None:
        log_document_start(
            doc_id=doc_id,
            doc_type=doc_type,
            pages=page_count,
            chars=len(raw_text),
            logger=logger,
        )

    # Build Hierarchical Node Tree
    tree = build_tree_from_markdown(raw_text)
    annotated = tree.to_annotated_text()
    index = tree.build_index()
    metadata["node_count"] = len(index) // 2

    return ParsedDocument(
        doc_id=doc_id,
        doc_type=doc_type,
        source_dataset=rec.get("source_dataset", "extractbench"),
        layout_text=annotated,
        raw_text=raw_text,
        page_count=page_count,
        metadata=metadata,
        nodes=index,
        node_tree=tree,
        tables=tables_data,
    )


def render_pdf_pages_base64(
    pdf_path: Path | str,
    max_pages: int = 10,
    dpi: int = 150,
    as_data_uri: bool = False,
) -> List[str]:
    """Render PDF pages (or load image files) as base64 strings.
    
    Args:
        pdf_path: Path to PDF document or image file (.png, .jpg, etc.).
        max_pages: Maximum number of pages to render (for multi-page PDFs).
        dpi: Resolution for PDF rasterization (default: 150 DPI).
        as_data_uri: If True, prefix with 'data:<mime>;base64,'. Default False.
        
    Returns:
        List of base64 strings (one per page or single element for an image).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}

    # Handle image files directly
    if ext in image_exts:
        img_bytes = path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        if as_data_uri:
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            return [f"data:{mime};base64,{b64}"]
        return [b64]

    images_b64: List[str] = []
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            fitz = None

    if fitz is not None:
        try:
            with fitz.open(str(path)) as doc:
                num_pages = min(len(doc), max_pages)
                for page_idx in range(num_pages):
                    page = doc[page_idx]
                    pix = page.get_pixmap(dpi=dpi)
                    png_bytes = pix.tobytes("png")
                    b64 = base64.b64encode(png_bytes).decode("utf-8")
                    if as_data_uri:
                        images_b64.append(f"data:image/png;base64,{b64}")
                    else:
                        images_b64.append(b64)
            return images_b64
        except Exception as e:
            logger.warning("pymupdf rendering failed for %s: %s", path, e)

    # Fallback to pdf2image
    try:
        from pdf2image import convert_from_path
        import io
        pages = convert_from_path(str(path), first_page=1, last_page=max_pages, dpi=dpi)
        for page in pages:
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            if as_data_uri:
                images_b64.append(f"data:image/png;base64,{b64}")
            else:
                images_b64.append(b64)
        return images_b64
    except Exception as e:
        logger.warning("pdf2image rendering fallback failed for %s: %s", path, e)

    return images_b64


