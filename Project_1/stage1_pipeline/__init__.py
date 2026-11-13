"""Stage 1 Document Identification, Layout-Aware Parsing, and Node Tree Pipeline."""

from stage1_pipeline.node_tree import Node, SectionNode, ParagraphNode, SentenceNode, TableNode, TableRowNode
from stage1_pipeline.parser import ParsedDocument, parse_document, identify_document_type, render_pdf_pages_base64
from stage1_pipeline.extractor import extract_document, Extractor
from stage1_pipeline.verifiers import verify_and_calibrate, VerifierConfig
from stage1_pipeline.router import (
    PipelineConfig,
    RouteDecision,
    determine_route,
    chunk_document_text,
    merge_chunk_predictions,
    split_schema,
    merge_schema_predictions,
)
from stage1_pipeline.logging_utils import (
    get_logger,
    setup_logger,
    log_document_start,
    log_route_decision,
    log_llm_request,
    log_llm_response,
    log_verification_detail,
    log_financial_gate,
    log_consensus_comparison,
    log_progress,
    log_stage_banner,
    Timer,
)

__all__ = [
    "Node",
    "SectionNode",
    "ParagraphNode",
    "SentenceNode",
    "TableNode",
    "TableRowNode",
    "ParsedDocument",
    "parse_document",
    "identify_document_type",
    "render_pdf_pages_base64",
    "extract_document",
    "Extractor",
    "verify_and_calibrate",
    "VerifierConfig",
    "PipelineConfig",
    "RouteDecision",
    "determine_route",
    "chunk_document_text",
    "merge_chunk_predictions",
    "split_schema",
    "merge_schema_predictions",
    "get_logger",
    "setup_logger",
    "log_document_start",
    "log_route_decision",
    "log_llm_request",
    "log_llm_response",
    "log_verification_detail",
    "log_financial_gate",
    "log_consensus_comparison",
    "log_progress",
    "log_stage_banner",
    "Timer",
]

