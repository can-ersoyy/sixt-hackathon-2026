"""Test script demonstrating and validating stage1_pipeline logging_utils."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stage1_pipeline.logging_utils import (
    Timer,
    get_logger,
    log_consensus_comparison,
    log_document_start,
    log_financial_gate,
    log_llm_request,
    log_llm_response,
    log_progress,
    log_route_decision,
    log_stage_banner,
    log_verification_detail,
    setup_logger,
)


def run_logging_tests():
    print("--- 1. Testing Default Logger & Banner ---")
    log_stage_banner("STAGE 1 PIPELINE OBSERVABILITY TEST", "Validating all structured event loggers")

    logger = get_logger("stage1_pipeline.test")

    print("\n--- 2. Testing Document Ingestion Logging ---")
    log_document_start(
        doc_id="short/07021-2016-p0014",
        doc_type="digital_pdf",
        pages=2,
        chars=8450,
        logger=logger,
    )
    log_document_start(
        doc_id="cuad/sample_contract",
        doc_type="CUAD_CONTRACT",
        pages=45,
        chars=142050,
        logger=logger,
    )

    print("\n--- 3. Testing Routing Decision Logging ---")
    log_route_decision(
        doc_id="short/07021-2016-p0014",
        modality="text",
        model="gpt-4o-mini",
        chunks=1,
        images=0,
        logger=logger,
    )
    log_route_decision(
        doc_id="tax_return/1040_scanned",
        modality="multimodal_vision",
        model="gpt-4o",
        chunks=2,
        images=2,
        logger=logger,
    )

    print("\n--- 4. Testing LLM Request & Response Logging ---")
    log_llm_request(
        model="gpt-4o-mini",
        prompt_chars=12400,
        chunk_idx=1,
        total_chunks=3,
        logger=logger,
    )

    with Timer("Mock API Call", logger=logger, level=20):  # 20 is INFO
        pass

    log_llm_response(
        model="gpt-4o-mini",
        latency_sec=1.28,
        status="200 OK",
        tokens={"prompt_tokens": 1450, "completion_tokens": 320, "total_tokens": 1770},
        logger=logger,
    )

    log_llm_response(
        model="gpt-4o",
        latency_sec=4.51,
        status="429 RateLimitExceeded",
        tokens=None,
        logger=logger,
    )

    print("\n--- 5. Testing Verification & Confidence Calibration Logging ---")
    # Grounded passing field
    log_verification_detail(
        field="Agreement Date",
        value="2021-04-12",
        raw_conf=0.92,
        calibrated_conf=0.99,
        reason="verbatim_grounded",
        evidence_span="sec-1/p-1/s-2",
        logger=logger,
    )
    # Questionable / hallucinated field
    log_verification_detail(
        field="Governing Law",
        value="State of Delaware",
        raw_conf=0.85,
        calibrated_conf=0.15,
        reason="not_in_evidence_span",
        evidence_span="sec-8/p-3",
        logger=logger,
    )

    print("\n--- 6. Testing Financial Consistency Gate Logging ---")
    # Passing gate
    log_financial_gate(
        status="PASS",
        expected=1250.00,
        actual=1250.00,
        diff=0.00,
        fields_affected=["subtotal", "tax_amount", "total_amount"],
        logger=logger,
    )
    # Failing gate
    log_financial_gate(
        status="FAIL",
        expected=5000.00,
        actual=4850.00,
        diff=150.00,
        fields_affected=["line_items_sum", "invoice_total"],
        logger=logger,
    )

    print("\n--- 7. Testing Multi-Model Consensus Logging ---")
    # Consensus match
    log_consensus_comparison(
        field="Parties",
        val1="Acme Holdings Inc.",
        val2="Acme Holdings Inc.",
        match=True,
        old_conf=0.88,
        new_conf=0.99,
        logger=logger,
    )
    # Consensus disagreement
    log_consensus_comparison(
        field="Renewal Term",
        val1="1 year automatic renewal",
        val2="Month to month",
        match=False,
        old_conf=0.80,
        new_conf=0.35,
        logger=logger,
    )

    print("\n--- 8. Testing Batch Progress Logging ---")
    log_progress(current=1, total=10, elapsed_sec=3.2, doc_id="doc_001", logger=logger)
    log_progress(current=5, total=10, elapsed_sec=16.8, doc_id="doc_005", logger=logger)
    log_progress(current=10, total=10, elapsed_sec=34.1, doc_id="doc_010", logger=logger)

    print("\n--- 9. Testing File Logging & Verbosity Configuration ---")
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        file_logger = setup_logger("file_test_logger", level="DEBUG", log_file=tmp_path)
        file_logger.debug("Debug event should appear in file.")
        log_document_start("doc_file_test", "tsv_table", pages=1, chars=500, logger=file_logger)

        with open(tmp_path, "r", encoding="utf-8") as f:
            file_contents = f.read()

        assert "doc_file_test" in file_contents, "File log content missing expected entry!"
        assert "Debug event" in file_contents, "Debug event missing in file log!"
        print(f"File log successfully written and verified ({len(file_contents)} bytes).")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n✅ All logging_utils tests completed successfully!")


if __name__ == "__main__":
    run_logging_tests()
