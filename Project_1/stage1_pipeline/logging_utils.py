"""Centralized logging & observability module for the Stage 1 Pipeline.

Provides clean, colorized terminal logging (via colorlog or ANSI fallback),
configurable verbosity (DEBUG, INFO, WARNING, ERROR), optional file logging,
and structured helper functions for all major pipeline stages:
  - Document parsing & type identification
  - Routing decisions (modality, model, chunking)
  - LLM requests & responses (latency, status, token usage)
  - Field verification & confidence calibration (grounding & span checks)
  - Financial/arithmetic sanity gates
  - Multi-run consensus comparisons
  - Pipeline batch progress & ETA calculation
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Try to import colorlog for rich terminal coloring; fallback to ANSI colors
try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

# ANSI Escape Sequences
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"
_BOLD_RED = "\033[1;31m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_YELLOW = "\033[1;33m"
_BOLD_BLUE = "\033[1;34m"
_BOLD_MAGENTA = "\033[1;35m"
_BOLD_CYAN = "\033[1;36m"
_BOLD_WHITE = "\033[1;37m"

# Default logger name
DEFAULT_LOGGER_NAME = "stage1_pipeline"


def _should_use_colors(stream: Any) -> bool:
    """Determine whether color output should be enabled for the given stream."""
    if os.environ.get("NO_COLOR") or os.environ.get("ANSI_COLORS_DISABLED"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


class _AnsiColorFormatter(logging.Formatter):
    """Fallback ANSI color formatter when colorlog is unavailable."""

    LEVEL_COLORS = {
        logging.DEBUG: _CYAN,
        logging.INFO: _GREEN,
        logging.WARNING: _YELLOW,
        logging.ERROR: _RED,
        logging.CRITICAL: _BOLD_RED,
    }

    def __init__(self, use_colors: bool = True):
        super().__init__(
            fmt="%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname
        orig_name = record.name
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, _WHITE)
            record.levelname = f"{color}{orig_levelname:<7}{_RESET}"
            record.name = f"{_DIM}{orig_name}{_RESET}"

        formatted = super().format(record)

        # Restore original record attributes to avoid side-effects on other handlers
        record.levelname = orig_levelname
        record.name = orig_name
        return formatted


class _PlainFormatter(logging.Formatter):
    """Plain text formatter for file logging or non-TTY outputs."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: Optional[Union[int, str]] = None,
    log_file: Optional[str] = None,
    stream: Any = sys.stderr,
) -> logging.Logger:
    """Configure and return a structured pipeline logger.

    Args:
        name: Logger name (e.g., 'stage1_pipeline' or child 'stage1_pipeline.parser').
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR') or int. If None,
               reads PIPELINE_LOG_LEVEL or LOG_LEVEL from the environment (default: INFO).
        log_file: Optional file path to also write clean uncolored logs to.
        stream: Output stream for console logging (defaults to sys.stderr).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Determine log level
    if level is None:
        env_level = os.environ.get("PIPELINE_LOG_LEVEL") or os.environ.get("LOG_LEVEL", "INFO")
        level = getattr(logging, env_level.upper(), logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Check if a custom stream handler is already attached to this specific logger
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    if not has_stream_handler:
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setLevel(level)
        use_colors = _should_use_colors(stream)

        if HAS_COLORLOG and use_colors:
            formatter = colorlog.ColoredFormatter(
                fmt="%(log_color)s%(asctime)s [%(levelname)-7s]%(reset)s %(blue)s[%(name)s]%(reset)s %(message)s",
                datefmt="%H:%M:%S",
                reset=True,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
                style="%",
            )
            stream_handler.setFormatter(formatter)
        else:
            stream_handler.setFormatter(_AnsiColorFormatter(use_colors=use_colors))

        logger.addHandler(stream_handler)

    # Optional file logging
    file_path = log_file or os.environ.get("PIPELINE_LOG_FILE")
    if file_path:
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(file_path)
            for h in logger.handlers
        )
        if not has_file_handler:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                file_handler = logging.FileHandler(file_path, encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(_PlainFormatter())
                logger.addHandler(file_handler)
            except OSError as err:
                logger.warning("Could not initialize log file %s: %s", file_path, err)

    # Don't propagate to root logger if already configured to avoid duplicate lines
    logger.propagate = False
    return logger


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Retrieve an existing logger or configure a default one if not yet initialized."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Global default logger instance
default_logger = get_logger(DEFAULT_LOGGER_NAME)


def _resolve_logger(custom_logger: Optional[logging.Logger]) -> logging.Logger:
    """Helper to resolve caller's logger or fallback to default_logger."""
    return custom_logger if custom_logger is not None else default_logger


# ============================================================================
# Structured Stage Event Loggers
# ============================================================================


def log_document_start(
    doc_id: str,
    doc_type: str,
    pages: Optional[int] = None,
    chars: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log document ingestion and initial parsing parameters.

    Example output:
      [DOC] 📄 doc_id=short/07021-2016-p0014 | type=digital_pdf | pages=1 | chars=6,197
    """
    log = _resolve_logger(logger)
    pages_str = str(pages) if pages is not None else "unknown"
    chars_str = f"{chars:,}" if chars is not None else "unknown"
    msg = f"[DOC] 📄 doc_id={doc_id} | type={doc_type} | pages={pages_str} | chars={chars_str}"
    log.info(msg)


def log_route_decision(
    doc_id: str,
    modality: str,
    model: str,
    chunks: int = 1,
    images: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log the routing decision made by router.py.

    Example output:
      [ROUTE] 🔀 doc_id=cuad/contract_01 | modality=text | model=gpt-4o-mini | chunks=3 | images=0
    """
    log = _resolve_logger(logger)
    msg = (
        f"[ROUTE] 🔀 doc_id={doc_id} -> modality={modality} | model={model} | "
        f"chunks={chunks} | images={images}"
    )
    log.info(msg)


def log_llm_request(
    model: str,
    prompt_chars: int,
    chunk_idx: int = 1,
    total_chunks: int = 1,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log an outgoing LLM extraction request.

    Example output:
      [LLM-REQ] 📤 model=gpt-4o-mini | chunk=1/3 | prompt_chars=28,450
    """
    log = _resolve_logger(logger)
    msg = (
        f"[LLM-REQ] 📤 model={model} | chunk={chunk_idx}/{total_chunks} | "
        f"prompt_chars={prompt_chars:,}"
    )
    log.info(msg)


def log_llm_response(
    model: str,
    latency_sec: float,
    status: str = "200 OK",
    tokens: Optional[Union[Dict[str, int], int]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log an LLM response with latency, status code, and token usage.

    Example output:
      [LLM-RESP] 📥 model=gpt-4o-mini | latency=1.24s | status=200 OK | tokens: prompt=1200, comp=350, total=1550
    """
    log = _resolve_logger(logger)
    if isinstance(tokens, dict):
        tok_str = (
            f"tokens: in={tokens.get('prompt_tokens', 0):,}, "
            f"out={tokens.get('completion_tokens', 0):,}, "
            f"total={tokens.get('total_tokens', 0):,}"
        )
    elif tokens is not None:
        tok_str = f"tokens={tokens:,}"
    else:
        tok_str = "tokens=N/A"

    msg = f"[LLM-RESP] 📥 model={model} | latency={latency_sec:.2f}s | status={status} | {tok_str}"
    if status.startswith("200") or status.upper() == "OK":
        log.info(msg)
    else:
        log.warning(msg)


def log_verification_detail(
    field: str,
    value: Any,
    raw_conf: float,
    calibrated_conf: float,
    reason: str,
    evidence_span: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log field verification outcome and confidence calibration.

    Example output:
      [VERIFY] 🔍 field='Agreement Date' | val='2021-04-12' | conf: 0.95 -> 0.98 | reason=verbatim_grounded [sec-1/p-1/s-2]
    """
    log = _resolve_logger(logger)

    # Format value snippet safely
    val_repr = repr(value)
    if len(val_repr) > 45:
        val_repr = val_repr[:42] + "..."

    span_str = f" [{evidence_span}]" if evidence_span else ""
    msg = (
        f"[VERIFY] 🔍 field='{field}' | val={val_repr} | "
        f"conf: {raw_conf:.2f} -> {calibrated_conf:.2f} | reason={reason}{span_str}"
    )

    if calibrated_conf >= 0.80:
        log.info(msg)
    elif calibrated_conf >= 0.30:
        log.warning(msg)
    else:
        log.info(msg)


def log_financial_gate(
    status: str,
    expected: Optional[Union[float, str]],
    actual: Optional[Union[float, str]],
    diff: Optional[Union[float, str]],
    fields_affected: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log an arithmetic/financial consistency check.

    Example output:
      [FIN-GATE] 💰 status=PASS | expected=1000.00 | actual=1000.00 | diff=0.00 | fields=['subtotal', 'tax', 'total']
    """
    log = _resolve_logger(logger)
    fields_str = f"fields={fields_affected}" if fields_affected else "fields=none"
    exp_str = f"{expected:.2f}" if isinstance(expected, (int, float)) else str(expected)
    act_str = f"{actual:.2f}" if isinstance(actual, (int, float)) else str(actual)
    diff_str = f"{diff:.2f}" if isinstance(diff, (int, float)) else str(diff)

    msg = (
        f"[FIN-GATE] 💰 status={status} | expected={exp_str} | "
        f"actual={act_str} | diff={diff_str} | {fields_str}"
    )

    if status.upper() in ("PASS", "OK", "MATCH"):
        log.info(msg)
    elif status.upper() in ("FAIL", "REJECT", "MISMATCH"):
        log.warning(msg)
    else:
        log.info(msg)


def log_consensus_comparison(
    field: str,
    val1: Any,
    val2: Any,
    match: bool,
    old_conf: float,
    new_conf: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log multi-model or multi-pass consensus comparison.

    Example output:
      [CONSENSUS] 🤝 field='Parties' | MATCH (✅) | conf: 0.85 -> 0.99 | v1='Acme Corp' | v2='Acme Corp'
    """
    log = _resolve_logger(logger)
    v1_repr = repr(val1)
    if len(v1_repr) > 30:
        v1_repr = v1_repr[:27] + "..."
    v2_repr = repr(val2)
    if len(v2_repr) > 30:
        v2_repr = v2_repr[:27] + "..."

    outcome = "MATCH (✅)" if match else "DIFF (❌)"
    msg = (
        f"[CONSENSUS] 🤝 field='{field}' | {outcome} | "
        f"conf: {old_conf:.2f} -> {new_conf:.2f} | v1={v1_repr} | v2={v2_repr}"
    )
    log.info(msg)


def log_progress(
    current: int,
    total: int,
    elapsed_sec: float,
    doc_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log batch processing progress, speed, and ETA.

    Example output:
      [PROGRESS] ⏳ [12/100] (12.0%) | elapsed=45.2s | eta=331.5s (3.77s/doc) | doc_id=short/07021-2016-p0014
    """
    log = _resolve_logger(logger)
    pct = (current / total * 100.0) if total > 0 else 0.0
    speed = (elapsed_sec / current) if current > 0 else 0.0
    remaining = max(0, total - current)
    eta = speed * remaining
    doc_str = f" | doc_id={doc_id}" if doc_id else ""

    msg = (
        f"[PROGRESS] ⏳ [{current}/{total}] ({pct:.1f}%) | "
        f"elapsed={elapsed_sec:.1f}s | eta={eta:.1f}s ({speed:.2f}s/doc){doc_str}"
    )
    log.info(msg)


def log_stage_banner(
    stage_name: str,
    description: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log an eye-catching visual section divider for major pipeline transitions."""
    log = _resolve_logger(logger)
    line = "=" * 65
    log.info(line)
    log.info("  🚀 %s", stage_name.upper())
    if description:
        log.info("     %s", description)
    log.info(line)


# ============================================================================
# Utilities
# ============================================================================


@contextlib.contextmanager
def Timer(operation_name: str, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """Context manager to measure and log block execution time.

    Example:
      with Timer("PDF OCR", logger=logger):
          run_ocr(...)
    """
    log = _resolve_logger(logger)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        log.log(level, f"[TIMER] ⏱️ {operation_name} completed in {elapsed:.3f}s")
