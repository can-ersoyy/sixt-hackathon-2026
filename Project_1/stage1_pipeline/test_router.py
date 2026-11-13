"""Unit tests for Stage 1 Router: Modality Selection, Document Chunking, and Schema Chunking."""
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stage1_pipeline.parser import ParsedDocument
from stage1_pipeline.router import (
    PipelineConfig,
    RouteDecision,
    determine_route,
    chunk_document_text,
    merge_chunk_predictions,
    split_schema,
    merge_schema_predictions,
)


class TestRouter(unittest.TestCase):

    def setUp(self):
        self.config_auto = PipelineConfig(
            route_mode="auto",
            vlm_max_pages=5,
            max_doc_chars=180_000,
            chunk_overlap_chars=5_000,
            schema_chunk_size=20,
        )

    def test_modality_selection_auto(self):
        # 1. CUAD Contract -> Text
        cuad_parsed = ParsedDocument(
            doc_id="cuad:contract_1",
            doc_type="cuad",
            source_dataset="cuad",
            layout_text="[sec-1] This is a legal contract.",
            raw_text="This is a legal contract.",
            page_count=1,
        )
        rec_cuad = {"doc_id": "cuad:contract_1", "source_dataset": "cuad"}
        decision = determine_route(rec_cuad, cuad_parsed, self.config_auto)
        self.assertEqual(decision.modality, "text")
        self.assertIn("CUAD", decision.reason)
        self.assertIsNone(decision.companion_text)

        # 2. Short digital PDF (1 page <= 5) -> VLM with companion text
        dig_parsed_short = ParsedDocument(
            doc_id="dig_short",
            doc_type="digital_pdf",
            source_dataset="extractbench",
            layout_text="[sec-1] Invoice text with table.",
            raw_text="Invoice text with table.",
            page_count=1,
        )
        rec_dig = {"doc_id": "dig_short", "pages": ["dummy.pdf"]}
        decision = determine_route(rec_dig, dig_parsed_short, self.config_auto)
        self.assertEqual(decision.modality, "vlm")
        self.assertEqual(decision.companion_text, "[sec-1] Invoice text with table.")
        self.assertIn("Short digital PDF", decision.reason)

        # 3. Long digital PDF (10 pages > 5) -> Text without companion text
        dig_parsed_long = ParsedDocument(
            doc_id="dig_long",
            doc_type="digital_pdf",
            source_dataset="extractbench",
            layout_text="[sec-1] Very long report...",
            raw_text="Very long report...",
            page_count=10,
        )
        decision = determine_route(rec_dig, dig_parsed_long, self.config_auto)
        self.assertEqual(decision.modality, "text")
        self.assertIsNone(decision.companion_text)
        self.assertIn("Long document", decision.reason)

        # 4. Scanned PDF (2 pages) -> VLM without companion text
        scan_parsed = ParsedDocument(
            doc_id="scanned_1",
            doc_type="scanned_pdf",
            source_dataset="extractbench",
            layout_text="[sec-1] OCR text",
            raw_text="OCR text",
            page_count=2,
        )
        decision = determine_route(rec_dig, scan_parsed, self.config_auto)
        self.assertEqual(decision.modality, "vlm")
        self.assertIsNone(decision.companion_text)
        self.assertIn("Scanned", decision.reason)

        # 5. Image doc -> VLM
        img_parsed = ParsedDocument(
            doc_id="img_1",
            doc_type="image",
            source_dataset="extractbench",
            layout_text="[sec-1] Image text",
            raw_text="Image text",
            page_count=1,
        )
        rec_img = {"doc_id": "img_1", "pages": ["page.png"]}
        decision = determine_route(rec_img, img_parsed, self.config_auto)
        self.assertEqual(decision.modality, "vlm")

    def test_modality_manual_overrides(self):
        cuad_parsed = ParsedDocument(
            doc_id="cuad:contract_1",
            doc_type="cuad",
            source_dataset="cuad",
            layout_text="[sec-1] Text",
            raw_text="Text",
            page_count=1,
        )
        rec = {"doc_id": "cuad:contract_1"}

        # Override to VLM
        cfg_vlm = PipelineConfig(route_mode="vlm")
        d_vlm = determine_route(rec, cuad_parsed, cfg_vlm)
        self.assertEqual(d_vlm.modality, "vlm")
        self.assertIn("route_mode='vlm'", d_vlm.reason)

        # Override to Text on short digital PDF
        dig_parsed = ParsedDocument(
            doc_id="dig_1",
            doc_type="digital_pdf",
            source_dataset="extractbench",
            layout_text="[sec-1] Short text",
            raw_text="Short text",
            page_count=1,
        )
        cfg_text = PipelineConfig(route_mode="text")
        d_text = determine_route({"doc_id": "dig_1"}, dig_parsed, cfg_text)
        self.assertEqual(d_text.modality, "text")
        self.assertIn("route_mode='text'", d_text.reason)

    def test_document_chunking(self):
        # Build a document with 10 sections of 1,000 chars each = 10,000 chars
        sections = [f"[sec-{i}] Section {i} content:\n" + ("x" * 950) for i in range(1, 11)]
        full_doc = "\n\n".join(sections)

        # Short doc (max_chars > len(full_doc)) -> single chunk
        single = chunk_document_text(full_doc, max_chars=20_000, overlap_chars=1_000)
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0], full_doc)

        # Long doc (max_chars = 3,500, overlap = 1,000) -> multiple chunks
        chunks = chunk_document_text(full_doc, max_chars=3_500, overlap_chars=1_000)
        self.assertGreater(len(chunks), 1)

        # Ensure each chunk contains section tags
        for c in chunks:
            self.assertIn("[sec-", c)

        # Ensure consecutive chunks overlap: text near the end of chunk 0 appears in chunk 1
        tail_snippet = chunks[0][-200:]
        self.assertTrue(any(snippet in chunks[1] for snippet in tail_snippet.split("\n") if len(snippet) > 20))

    def test_merge_chunk_predictions(self):
        chunk_0 = {
            "vendor_name": {"value": "Acme Corp", "confidence": 0.70, "evidence_span": "[sec-1]"},
            "invoice_total": {"value": None, "confidence": 0.0},
            "is_tax_exempt": {"value": False, "confidence": 0.90},  # Boolean False is non-null!
        }
        chunk_1 = {
            "vendor_name": {"value": "Acme Corporation International", "confidence": 0.95, "evidence_span": "[sec-4]"},
            "invoice_total": {"value": "1250.00", "confidence": 0.88, "evidence_span": "[sec-5]"},
            "is_tax_exempt": {"value": None, "confidence": 0.0},
        }
        chunk_2 = {
            "vendor_name": {"value": "Acme", "confidence": 0.50},
            "invoice_total": {"value": "1250.00", "confidence": 0.60},
            "unseen_field": {"value": None, "confidence": 0.0},
        }

        merged = merge_chunk_predictions([chunk_0, chunk_1, chunk_2])

        # vendor_name picks highest confidence (0.95 from chunk_1)
        self.assertEqual(merged["vendor_name"]["value"], "Acme Corporation International")
        self.assertEqual(merged["vendor_name"]["confidence"], 0.95)
        self.assertEqual(merged["vendor_name"]["evidence_span"], "[sec-4]")

        # invoice_total was null in chunk_0, picked 0.88 from chunk_1
        self.assertEqual(merged["invoice_total"]["value"], "1250.00")
        self.assertEqual(merged["invoice_total"]["confidence"], 0.88)

        # is_tax_exempt preserved boolean False with 0.90
        self.assertEqual(merged["is_tax_exempt"]["value"], False)
        self.assertEqual(merged["is_tax_exempt"]["confidence"], 0.90)

        # unseen_field was null in all chunks -> null with 0.0
        self.assertIsNone(merged["unseen_field"]["value"])
        self.assertEqual(merged["unseen_field"]["confidence"], 0.0)

    def test_schema_chunking(self):
        # Create schema with 45 properties
        properties = {f"field_{i}": {"type": "string", "description": f"Field {i}"} for i in range(1, 46)}
        schema = {
            "$defs": {"some_def": {}},
            "title": "DenseTaxForm",
            "type": "object",
            "properties": properties,
            "required": ["field_1", "field_25", "field_45"],
        }

        sub_schemas = split_schema(schema, chunk_size=20)
        self.assertEqual(len(sub_schemas), 3)

        # Chunk 0 has 20 props, chunk 1 has 20 props, chunk 2 has 5 props
        self.assertEqual(len(sub_schemas[0]["properties"]), 20)
        self.assertEqual(len(sub_schemas[1]["properties"]), 20)
        self.assertEqual(len(sub_schemas[2]["properties"]), 5)

        # Preserved metadata and definitions
        for s in sub_schemas:
            self.assertEqual(s["title"], "DenseTaxForm")
            self.assertEqual(s["type"], "object")
            self.assertIn("some_def", s["$defs"])

        # Required fields correctly partitioned
        self.assertEqual(sub_schemas[0]["required"], ["field_1"])
        self.assertEqual(sub_schemas[1]["required"], ["field_25"])
        self.assertEqual(sub_schemas[2]["required"], ["field_45"])

        # Schema with <= chunk_size returns single schema
        small_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assertEqual(len(split_schema(small_schema, chunk_size=20)), 1)

    def test_merge_schema_predictions(self):
        sub_0 = {
            "fields": {
                "field_1": {"value": "Val 1", "confidence": 0.95},
                "field_2": {"value": "Val 2", "confidence": 0.90},
            }
        }
        sub_1 = {
            "field_3": {"value": "Val 3", "confidence": 0.85},
            "field_4": {"value": None, "confidence": 0.0},
        }

        merged = merge_schema_predictions([sub_0, sub_1])
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged["field_1"]["value"], "Val 1")
        self.assertEqual(merged["field_2"]["value"], "Val 2")
        self.assertEqual(merged["field_3"]["value"], "Val 3")
        self.assertIsNone(merged["field_4"]["value"])

    def test_real_pdf_vlm_routing(self):
        # Test routing with real PDF file on disk
        pdf_rel = "raw/extractbench/docs/short/07021-2016-p0014.pdf"
        rec = {"doc_id": "test_real_pdf", "pages": [pdf_rel]}
        parsed = ParsedDocument(
            doc_id="test_real_pdf",
            doc_type="digital_pdf",
            source_dataset="extractbench",
            layout_text="[sec-1] Real PDF Layout Text",
            raw_text="Real PDF text",
            page_count=1,
        )

        decision = determine_route(rec, parsed, self.config_auto)
        self.assertEqual(decision.modality, "vlm")
        self.assertEqual(len(decision.images_b64), 1)
        self.assertGreater(len(decision.images_b64[0]), 1000)
        self.assertEqual(decision.companion_text, "[sec-1] Real PDF Layout Text")


if __name__ == "__main__":
    unittest.main()
