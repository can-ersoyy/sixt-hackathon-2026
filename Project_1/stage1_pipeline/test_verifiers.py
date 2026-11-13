"""Comprehensive Unit Tests for Stage 1 Verifiers and Calibration Module.

Tests:
1. Checkbox handling (exempting False / false from raw-text absence penalties).
2. Grounding check on node neighbors (neighbor window and fuzzy matching).
3. Financial arithmetic gate (subtotal + tax - discount == total check).
4. CUAD multi-span check (splitting by " | " and verifying alignment against raw text).
5. Risk multipliers (JSON all-or-nothing, scanned/corrupted doc penalties).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stage1_pipeline.node_tree import build_tree_from_markdown
from stage1_pipeline.parser import ParsedDocument
from stage1_pipeline.verifiers import (
    VerifierConfig,
    verify_and_calibrate,
    norm_string,
    norm_id,
    norm_amount,
    norm_date,
    norm_json,
    span_f1,
    values_match,
    normalize,
)


def _create_mock_parsed_doc(
    doc_id: str = "test_doc_01",
    doc_type: str = "digital_pdf",
    source_dataset: str = "extractbench",
    raw_text: str = "",
) -> ParsedDocument:
    """Helper to create a fully initialized ParsedDocument with node tree."""
    tree = build_tree_from_markdown(raw_text)
    annotated = tree.to_annotated_text()
    index = tree.build_index()

    return ParsedDocument(
        doc_id=doc_id,
        doc_type=doc_type,
        source_dataset=source_dataset,
        layout_text=annotated,
        raw_text=raw_text,
        page_count=1,
        metadata={"char_count": len(raw_text)},
        nodes=index,
        node_tree=tree,
        tables=[],
    )


def test_normalizers_consistency():
    """Verify normalizers are imported and match expected behavior."""
    assert norm_string("  Hello   WORLD \n") == "hello world"
    assert norm_id("INV-2023/001-A") == "inv2023001a"
    assert norm_amount("$1,234.50") == "1234.50"
    assert norm_amount("1.234,50 €") == "1234.50"
    assert norm_date("2023-05-15") == "2023-05-15"
    assert norm_date("15/05/2023") == "2023-05-15"
    assert norm_json('{"b": 2, "a": 1}') == '{"a":1,"b":2}'
    assert values_match("1,234.50", "1234.5", "amount") is True


def test_checkbox_handling():
    """Test checkbox exception: unchecked boxes must not be penalized for lacking text in doc."""
    raw_doc = (
        "Applicant: Jane Doe\n"
        "Date of Birth: 1985-04-12\n"
        "Income: $55,000.00\n"
    )
    parsed = _create_mock_parsed_doc(raw_text=raw_doc)
    rec = {
        "doc_id": "doc_form_01",
        "tags": [],
        "schema": {
            "properties": {
                "has_prior_convictions": {
                    "type": "boolean",
                    "description": "Whether applicant checked prior convictions box",
                },
                "is_us_citizen": {
                    "type": "boolean",
                    "description": "Whether applicant is a US citizen",
                },
                "applicant_name": {
                    "type": "string",
                    "x-normalizer": "string",
                },
            }
        },
    }

    fields = {
        "has_prior_convictions": {
            "value": False,
            "confidence": 0.95,
            "evidence_span": None,
            "quote": None,
        },
        "is_us_citizen": {
            "value": "false",
            "confidence": 0.92,
            "evidence_span": None,
            "quote": None,
        },
        "applicant_name": {
            "value": "Jane Doe",
            "confidence": 0.95,
            "evidence_span": None,
            "quote": "Jane Doe",
        },
    }

    calibrated = verify_and_calibrate(rec, parsed, fields)

    # Both false checkboxes must maintain their confidence without ungrounded 0.20 drop
    assert calibrated["has_prior_convictions"]["confidence"] >= 0.90
    assert calibrated["is_us_citizen"]["confidence"] >= 0.90
    assert calibrated["applicant_name"]["confidence"] >= 0.95


def test_node_neighbor_grounding():
    """Test neighbor node retrieval and fuzzy matching across node boundaries."""
    doc_markdown = (
        "### Section 1 Agreement\n\n"
        "This agreement is entered into by and between Acme Corp and Beta LLC. "
        "The effective date shall be October 15, 2024. "
        "All terms and conditions herein apply."
    )
    parsed = _create_mock_parsed_doc(raw_text=doc_markdown)

    # Locate sentence node path
    sentence_nodes = [k for k in parsed.nodes if "s-" in k and k.startswith("[")]
    assert len(sentence_nodes) >= 2, f"Expected sentence nodes, got {parsed.nodes.keys()}"
    span_s1 = sentence_nodes[0]
    span_s2 = sentence_nodes[1]

    rec = {
        "doc_id": "contract_neighbor_test",
        "tags": [],
        "schema": {
            "properties": {
                "parties": {"type": "string", "x-normalizer": "string"},
                "effective_date": {"type": "string", "x-normalizer": "date"},
                "hallucinated_field": {"type": "string", "x-normalizer": "string"},
            }
        },
    }

    # Neighbor match: effective date is in s2, but evidence span cited is s1 (immediate neighbor)
    fields = {
        "effective_date": {
            "value": "2024-10-15",
            "confidence": 0.94,
            "evidence_span": span_s1,  # neighbor node!
            "quote": "October 15, 2024",
        },
        "parties": {
            "value": "Acme Corp and Beta LLC",
            "confidence": 0.96,
            "evidence_span": span_s1,
            "quote": "Acme Corp and Beta LLC",
        },
        "hallucinated_field": {
            "value": "Non-existent Unicorn Corp",
            "confidence": 0.95,
            "evidence_span": span_s1,
            "quote": "Unicorn Corp",
        },
    }

    calibrated = verify_and_calibrate(rec, parsed, fields)

    # Effective date neighbor grounding succeeds via get_node_with_neighbors
    assert calibrated["effective_date"]["confidence"] >= 0.85
    # Parties exact node grounding succeeds
    assert calibrated["parties"]["confidence"] >= 0.90
    # Hallucinated value completely ungrounded drops by 0.20x
    assert calibrated["hallucinated_field"]["confidence"] <= (0.95 * 0.20 + 0.05)


def test_financial_arithmetic_gate_pass():
    """Test financial arithmetic gate: subtotal + tax - discount == total boosts confidence."""
    raw_invoice = (
        "INVOICE\n"
        "Subtotal: $1,000.00\n"
        "Sales Tax (8%): $80.00\n"
        "Discount Coupon: $30.00\n"
        "Total Amount Due: $1,050.00\n"
    )
    parsed = _create_mock_parsed_doc(raw_text=raw_invoice)
    rec = {
        "doc_id": "inv_pass_01",
        "tags": [],
        "schema": {
            "properties": {
                "subtotal": {"type": "number", "x-normalizer": "amount"},
                "tax": {"type": "number", "x-normalizer": "amount"},
                "discount_amount": {"type": "number", "x-normalizer": "amount"},
                "total": {"type": "number", "x-normalizer": "amount"},
            }
        },
    }

    fields = {
        "subtotal": {"value": "1000.00", "confidence": 0.90, "evidence_span": None, "quote": "$1,000.00"},
        "tax": {"value": "80.00", "confidence": 0.90, "evidence_span": None, "quote": "$80.00"},
        "discount_amount": {"value": "30.00", "confidence": 0.88, "evidence_span": None, "quote": "$30.00"},
        "total": {"value": "1050.00", "confidence": 0.91, "evidence_span": None, "quote": "$1,050.00"},
    }

    calibrated = verify_and_calibrate(rec, parsed, fields)

    # 1000 + 80 - 30 == 1050 (diff 0.00 <= 0.02) -> financial boost applied (+0.03)
    assert abs(calibrated["subtotal"]["confidence"] - 0.93) < 0.005
    assert abs(calibrated["tax"]["confidence"] - 0.93) < 0.005
    assert abs(calibrated["discount_amount"]["confidence"] - 0.91) < 0.005
    assert abs(calibrated["total"]["confidence"] - 0.94) < 0.005


def test_financial_arithmetic_gate_fail():
    """Test financial arithmetic gate: disagreement penalizes confidence by 0.50x."""
    raw_invoice = (
        "INVOICE\n"
        "Subtotal: $1,000.00\n"
        "Sales Tax: $80.00\n"
        "Total: $1,500.00\n"
    )
    parsed = _create_mock_parsed_doc(raw_text=raw_invoice)
    rec = {
        "doc_id": "inv_fail_01",
        "tags": [],
        "schema": {
            "properties": {
                "subtotal": {"type": "number", "x-normalizer": "amount"},
                "tax": {"type": "number", "x-normalizer": "amount"},
                "total": {"type": "number", "x-normalizer": "amount"},
            }
        },
    }

    fields = {
        "subtotal": {"value": "1000.00", "confidence": 0.90, "evidence_span": None, "quote": "$1,000.00"},
        "tax": {"value": "80.00", "confidence": 0.90, "evidence_span": None, "quote": "$80.00"},
        "total": {"value": "1500.00", "confidence": 0.90, "evidence_span": None, "quote": "$1,500.00"},
    }

    calibrated = verify_and_calibrate(rec, parsed, fields)

    # Expected: 1000 + 80 = 1080. Total is 1500. Diff = 420.00 > 0.02 -> 0.50x penalty
    assert abs(calibrated["subtotal"]["confidence"] - 0.45) < 0.01
    assert abs(calibrated["tax"]["confidence"] - 0.45) < 0.01
    assert abs(calibrated["total"]["confidence"] - 0.45) < 0.01


def test_cuad_multi_span_check():
    """Test CUAD multi-span check with ' | ' split and alignment to raw text."""
    contract_text = (
        "EXHIBIT 10.1\n\n"
        "NON-COMPETE AGREEMENT\n\n"
        "Section 5. Non-Competition.\n"
        "Employee covenants that for a period of two (2) years following termination, "
        "Employee shall not engage in any competitive business in North America.\n\n"
        "Section 6. Governing Law.\n"
        "This Agreement shall be construed in accordance with the laws of the State of Delaware."
    )
    parsed = _create_mock_parsed_doc(doc_type="cuad", source_dataset="cuad", raw_text=contract_text)
    rec = {
        "doc_id": "cuad:TEST_CONTRACT",
        "tags": [],
        "schema": {
            "properties": {
                "Non-Compete": {"type": "string", "x-normalizer": "span"},
                "Governing Law": {"type": "string", "x-normalizer": "span"},
                "Audit Rights": {"type": "string", "x-normalizer": "span"},
            }
        },
    }

    # Case 1: Multi-span where both segments align
    val_multi_good = (
        "Employee covenants that for a period of two (2) years following termination | "
        "Employee shall not engage in any competitive business in North America"
    )
    # Case 2: Multi-span where one segment is completely fake
    val_multi_partial = (
        "This Agreement shall be construed in accordance with the laws of the State of Delaware | "
        "The company shall grant stock options worth five billion dollars immediately"
    )
    # Case 3: Completely ungrounded span
    val_hallucinated = "The Executive shall receive a private jet for all personal travels."

    fields = {
        "Non-Compete": {"value": val_multi_good, "confidence": 0.95, "evidence_span": None, "quote": None},
        "Governing Law": {"value": val_multi_partial, "confidence": 0.95, "evidence_span": None, "quote": None},
        "Audit Rights": {"value": val_hallucinated, "confidence": 0.90, "evidence_span": None, "quote": None},
    }

    calibrated = verify_and_calibrate(rec, parsed, fields)

    # Multi-span good: both clauses match -> high confidence preserved
    assert calibrated["Non-Compete"]["confidence"] >= 0.90
    # Multi-span partial: scaled down proportionally
    assert 0.40 <= calibrated["Governing Law"]["confidence"] <= 0.65
    # Completely ungrounded span: dropped by 0.20x
    assert calibrated["Audit Rights"]["confidence"] <= (0.90 * 0.20 + 0.05)


def test_risk_multipliers():
    """Test configurable risk multipliers on JSON fields and scanned/corrupted documents."""
    parsed_scanned = _create_mock_parsed_doc(
        doc_type="scanned_pdf",
        raw_text="Invoice 12345 for John Smith dated 2023-01-01 total 50.00"
    )
    rec = {
        "doc_id": "scanned_doc_corrupted",
        "tags": ["corrupted"],
        "schema": {
            "properties": {
                "line_items": {"type": "array", "x-normalizer": "json"},
                "invoice_id": {"type": "string", "x-normalizer": "id"},
            }
        },
    }

    fields = {
        "line_items": {
            "value": '[{"item": "widget", "price": 50.0}]',
            "confidence": 0.90,
            "evidence_span": None,
            "quote": None,
        },
        "invoice_id": {
            "value": "12345",
            "confidence": 0.90,
            "evidence_span": None,
            "quote": "12345",
        },
    }

    # Custom config with distinct multipliers
    cfg = VerifierConfig(risk_json_mult=0.90, risk_corrupted_mult=0.85)
    calibrated = verify_and_calibrate(rec, parsed_scanned, fields, config=cfg)

    # line_items has both corrupted (0.85) and json (0.90) risk multipliers:
    # 0.90 * 0.90 * 0.85 = 0.6885
    assert abs(calibrated["line_items"]["confidence"] - 0.6885) < 0.01

    # invoice_id has only corrupted (0.85) risk multiplier:
    # 0.90 * 0.85 = 0.765
    assert abs(calibrated["invoice_id"]["confidence"] - 0.765) < 0.01


if __name__ == "__main__":
    print("Running verifiers tests directly...")
    test_normalizers_consistency()
    print("  [OK] Normalizers consistency")
    test_checkbox_handling()
    print("  [OK] Checkbox handling")
    test_node_neighbor_grounding()
    print("  [OK] Node neighbor grounding")
    test_financial_arithmetic_gate_pass()
    print("  [OK] Financial arithmetic gate pass")
    test_financial_arithmetic_gate_fail()
    print("  [OK] Financial arithmetic gate fail")
    test_cuad_multi_span_check()
    print("  [OK] CUAD multi-span check")
    test_risk_multipliers()
    print("  [OK] Risk multipliers")
    print("ALL TESTS PASSED SUCCESSFULLY!")
