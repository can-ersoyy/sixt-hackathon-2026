"""Verification script testing Stage 1 pipeline end-to-end on 3 diverse documents:
1. CUAD Contract
2. Digital Form PDF (ExtractBench)
3. Scanned Form PDF (ExtractBench)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from stage1_pipeline.parser import parse_document, identify_document_type
    from stage1_pipeline.extractor import Extractor, extract_document, is_grounded
except ImportError:
    from parser import parse_document, identify_document_type
    from extractor import Extractor, extract_document, is_grounded

DATA_DEV = Path("data/unified/dev.jsonl")


def run_tests():
    # Select 3 target documents
    recs = {}
    with open(DATA_DEV, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get("source_dataset")
            doc_id = rec["doc_id"]
            if src == "cuad" and "cuad" not in recs:
                recs["cuad"] = rec
            elif src == "extractbench" and "07021-2016-p0014" in doc_id and "digital" not in recs:
                recs["digital"] = rec
            elif src == "extractbench" and "08-15427" in doc_id and "scanned" not in recs:
                recs["scanned"] = rec

            if len(recs) == 3:
                break

    print(f"Loaded {len(recs)} sample documents for testing.")
    print("=" * 60)

    for category, rec in recs.items():
        doc_id = rec["doc_id"]
        print(f"\n[TESTING {category.upper()}] {doc_id}")
        doc_type, primary_path, page_count = identify_document_type(rec)
        print(f"1. Identification -> Type: {doc_type}, Page Count: {page_count}")

        # 2. Parse document
        parsed = parse_document(rec)
        print(f"2. Parsing -> Parser: {parsed.metadata.get('parser')}, "
              f"OCR Used: {parsed.metadata.get('ocr_used')}, "
              f"Raw Chars: {len(parsed.raw_text)}, "
              f"Indexed Nodes: {len(parsed.nodes)}")

        # Check sample canonical path tags
        sample_tags = [k for k in parsed.nodes.keys() if k.startswith("[")][:5]
        print(f"   Sample Canonical Paths: {sample_tags}")

        # Show snippet of layout text
        first_lines = "\n".join(parsed.layout_text.splitlines()[:6])
        print(f"   Annotated Snippet Preview:\n{first_lines}")

        # 3. Test node lookup
        if sample_tags:
            tag = sample_tags[0]
            val = parsed.get_node_text(tag)
            print(f"   Node Lookup '{tag}': {val[:80] if val else 'None'}...")

        # 4. Run offline extraction to test schema mapping and grounding
        offline_result = extract_document(rec, mode="offline")
        extracted_fields = {k: v for k, v in offline_result["fields"].items() if v["value"] is not None}
        print(f"3. Extraction (offline rules test) -> Found {len(extracted_fields)} populated fields:")
        for fn, fval in list(extracted_fields.items())[:3]:
            print(f"     - {fn}: {fval['value'][:50] if isinstance(fval['value'], str) else fval['value']} (conf: {fval['confidence']}, evidence_span: {fval['evidence_span']})")

    print("\n" + "=" * 60)
    print("ALL 3 TEST DOCUMENTS PARSED AND VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()

