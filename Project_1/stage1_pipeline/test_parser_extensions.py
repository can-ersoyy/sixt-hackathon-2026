"""Unit tests for Stage 1 Parser Extensions:
1. render_pdf_pages_base64 (PDF page rendering and image handling)
2. ParsedDocument.get_node_with_neighbors and Node.find_node_by_path
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stage1_pipeline.node_tree import (
    Node,
    SectionNode,
    ParagraphNode,
    SentenceNode,
    TableNode,
    TableRowNode,
    build_tree_from_markdown,
)
from stage1_pipeline.parser import (
    ParsedDocument,
    parse_document,
    render_pdf_pages_base64,
)

SAMPLE_PDF = Path("data/raw/extractbench/docs/short/07021-2016-p0014.pdf")


class TestRenderPdfPagesBase64(unittest.TestCase):
    """Test render_pdf_pages_base64 on PDFs and image files."""

    def test_render_pdf_pages_base64_real_pdf(self):
        self.assertTrue(SAMPLE_PDF.exists(), f"Sample PDF missing: {SAMPLE_PDF}")
        pages = render_pdf_pages_base64(SAMPLE_PDF, max_pages=2, dpi=150)
        self.assertIsInstance(pages, list)
        self.assertGreaterEqual(len(pages), 1)

        # Verify page content is valid base64-encoded PNG
        first_page = pages[0]
        self.assertIsInstance(first_page, str)
        decoded = base64.b64decode(first_page)
        # PNG signature: 89 50 4E 47 0D 0A 1A 0A
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_render_pdf_pages_base64_data_uri(self):
        pages = render_pdf_pages_base64(SAMPLE_PDF, max_pages=1, as_data_uri=True)
        self.assertEqual(len(pages), 1)
        self.assertTrue(pages[0].startswith("data:image/png;base64,"))
        raw_b64 = pages[0].split("data:image/png;base64,")[1]
        decoded = base64.b64decode(raw_b64)
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_render_pdf_pages_base64_max_pages_limit(self):
        pages = render_pdf_pages_base64(SAMPLE_PDF, max_pages=1)
        self.assertEqual(len(pages), 1)

    def test_render_image_file_directly(self):
        # Create a small dummy PNG file
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            tmp_path = Path(f.name)

        try:
            # Test direct image loading
            b64_list = render_pdf_pages_base64(tmp_path)
            self.assertEqual(len(b64_list), 1)
            decoded = base64.b64decode(b64_list[0])
            self.assertEqual(decoded, png_bytes)

            # Test direct image with as_data_uri=True
            uri_list = render_pdf_pages_base64(tmp_path, as_data_uri=True)
            self.assertEqual(len(uri_list), 1)
            self.assertTrue(uri_list[0].startswith("data:image/png;base64,"))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_render_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            render_pdf_pages_base64("nonexistent_file_path_12345.pdf")


class TestGetNodeWithNeighbors(unittest.TestCase):
    """Test ParsedDocument.get_node_with_neighbors and Node tree navigation."""

    def setUp(self):
        # Synthetic document with sections, paragraphs, sentences, and a table
        text = (
            "# Section 1\n\n"
            "This is sentence one. This is sentence two. This is sentence three.\n\n"
            "Standalone sentence.\n\n"
            "| Col A | Col B |\n"
            "|---|---|\n"
            "| Row 1A | Row 1B |\n"
            "| Row 2A | Row 2B |\n"
            "| Row 3A | Row 3B |\n"
        )
        self.tree = build_tree_from_markdown(text)
        self.annotated = self.tree.to_annotated_text()
        self.nodes_index = self.tree.build_index()
        self.parsed = ParsedDocument(
            doc_id="test_doc_001",
            doc_type="cuad",
            source_dataset="test",
            layout_text=self.annotated,
            raw_text=text,
            page_count=1,
            nodes=self.nodes_index,
            node_tree=self.tree,
        )

    def test_middle_sentence_neighbors(self):
        # Middle sentence s-2 in p-1 has s-1 and s-3 as siblings
        s2_text = self.parsed.get_node_text("[sec-1/p-1/s-2]")
        self.assertEqual(s2_text, "This is sentence two.")

        combined = self.parsed.get_node_with_neighbors("[sec-1/p-1/s-2]")
        # Sentences are joined with spaces
        expected = "This is sentence one. This is sentence two. This is sentence three."
        self.assertEqual(combined, expected)

        # Test without brackets
        combined_nobrackets = self.parsed.get_node_with_neighbors("sec-1/p-1/s-2")
        self.assertEqual(combined_nobrackets, expected)

    def test_first_sentence_neighbors(self):
        # First sentence s-1 has no previous sibling, only next sibling s-2
        combined = self.parsed.get_node_with_neighbors("[sec-1/p-1/s-1]")
        expected = "This is sentence one. This is sentence two."
        self.assertEqual(combined, expected)

    def test_last_sentence_neighbors(self):
        # Last sentence s-3 has previous sibling s-2, no next sibling
        combined = self.parsed.get_node_with_neighbors("[sec-1/p-1/s-3]")
        expected = "This is sentence two. This is sentence three."
        self.assertEqual(combined, expected)

    def test_standalone_node_without_siblings(self):
        # p-2 has only 1 sentence: s-1. Should fall back gracefully to its own text
        standalone_text = self.parsed.get_node_text("[sec-1/p-2/s-1]")
        self.assertEqual(standalone_text, "Standalone sentence.")

        combined = self.parsed.get_node_with_neighbors("[sec-1/p-2/s-1]")
        self.assertEqual(combined, standalone_text)

    def test_table_row_neighbors(self):
        # Table rows should be joined with newlines
        # row-0 is header, row-2 is Row 1A, row-3 is Row 2A, row-4 is Row 3A
        combined = self.parsed.get_node_with_neighbors("[sec-1/table-0/row-3]")
        self.assertIn("Row 1A", combined)
        self.assertIn("Row 2A", combined)
        self.assertIn("Row 3A", combined)
        self.assertIn("\n", combined)

    def test_fallback_on_missing_tree_or_path(self):
        # Nonexistent path
        empty_res = self.parsed.get_node_with_neighbors("[sec-99/p-99]")
        self.assertEqual(empty_res, "")

        # ParsedDocument with node_tree=None should fall back to self.get_node_text
        doc_no_tree = ParsedDocument(
            doc_id="test_fallback",
            doc_type="cuad",
            source_dataset="test",
            layout_text="[sec-1/p-1/s-1] fallback content",
            raw_text="fallback content",
            page_count=1,
            nodes={"[sec-1/p-1/s-1]": "fallback content", "sec-1/p-1/s-1": "fallback content"},
            node_tree=None,
        )
        self.assertEqual(doc_no_tree.get_node_with_neighbors("[sec-1/p-1/s-1]"), "fallback content")

    def test_on_real_cuad_document(self):
        # Verify on real CUAD document from dev.jsonl
        dev_jsonl = Path("data/unified/dev.jsonl")
        self.assertTrue(dev_jsonl.exists())
        with open(dev_jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source_dataset") == "cuad":
                    parsed = parse_document(rec)
                    # Find a path with s-2
                    for path in parsed.nodes:
                        if "/s-2" in path and path.startswith("["):
                            res = parsed.get_node_with_neighbors(path)
                            self.assertIsInstance(res, str)
                            self.assertGreater(len(res), len(parsed.get_node_text(path) or ""))
                            return
        self.fail("No CUAD record with multi-sentence paragraph found")


if __name__ == "__main__":
    unittest.main()
