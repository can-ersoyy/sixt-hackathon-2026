"""Stage 1 Verifiers and Confidence Calibration Module.

Implements verification and confidence calibration logic described in test.md (Section 3):
1. Exact normalizer reuse from 5_score.py.
2. Checkbox Exception (preserves unchecked boolean confidence without requiring text evidence).
3. Normalized & Fuzzy Grounding against node neighbor window and raw text.
4. CUAD multi-span alignment checks (split by " | " with partial ratio).
5. Financial Arithmetic Gate (subtotal + tax - discount == total).
6. Configurable Risk Multipliers for high-risk slices (JSON, corrupted/scanned docs).
"""
from __future__ import annotations

import importlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root and package directory are in sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 1. Import normalizers directly from 5_score.py
try:
    _score_mod = importlib.import_module("5_score")
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd()))
    _score_mod = importlib.import_module("5_score")

norm_string = _score_mod.norm_string
norm_id = _score_mod.norm_id
norm_amount = _score_mod.norm_amount
norm_date = _score_mod.norm_date
norm_json = _score_mod.norm_json
span_f1 = _score_mod.span_f1
values_match = _score_mod.values_match
normalize = _score_mod.normalize

# RapidFuzz for fast fuzzy matching
try:
    import rapidfuzz.fuzz as fuzz
except ImportError:
    class _DummyFuzz:
        @staticmethod
        def partial_ratio(s1: str, s2: str) -> float:
            s1_l, s2_l = s1.lower(), s2.lower()
            return 100.0 if (s1_l in s2_l or s2_l in s1_l) else 0.0
    fuzz = _DummyFuzz()

# Pipeline dependencies
try:
    from stage1_pipeline.parser import ParsedDocument
except ImportError:
    from parser import ParsedDocument

try:
    from stage1_pipeline.logging_utils import (
        get_logger,
        log_verification_detail,
        log_financial_gate,
    )
    logger = get_logger("stage1_pipeline.verifiers")
except ImportError:
    logger = logging.getLogger("stage1_pipeline.verifiers")
    log_verification_detail = None
    log_financial_gate = None


@dataclass
class VerifierConfig:
    """Configuration thresholds and multipliers for verification & calibration."""
    risk_json_mult: float = 0.90
    risk_corrupted_mult: float = 0.85
    grounding_fuzzy_threshold: float = 75.0
    cuad_span_fuzzy_threshold: float = 80.0
    ungrounded_penalty: float = 0.20
    financial_boost: float = 0.03
    financial_penalty: float = 0.50
    financial_tol: float = 0.02
    financial_max_boost: float = 0.995


def _get_cfg(config: Any, key: str, default: Any) -> Any:
    """Helper to retrieve configuration value from dict, dataclass, or fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get") and callable(config.get):
        return config.get(key, default)
    return getattr(config, key, default)


def _parse_amount(v: Any) -> Optional[float]:
    """Parse amount string or number to float using 5_score's norm_amount."""
    if v is None or v == "":
        return None
    try:
        if isinstance(v, (int, float)):
            return float(v)
        s = norm_amount(str(v))
        if s:
            return float(s)
    except (ValueError, TypeError):
        pass
    return None


SUBTOTAL_CANDIDATES = ("subtotal", "sub_total", "net_amount", "net", "sub_total_amount")
TAX_CANDIDATES = ("tax", "vat_amount", "vat", "tax_amount", "sales_tax", "taxes")
DISCOUNT_CANDIDATES = ("discount_amount", "discount", "discounts", "discount_val", "rebate")
TOTAL_CANDIDATES = ("total", "total_amount", "gross_amount", "invoice_total", "amount_due", "grand_total")


def _match_field_name(fields: Dict[str, dict], candidates: Tuple[str, ...]) -> Optional[str]:
    """Finds field name in fields matching candidate names."""
    # Exact match first
    for cand in candidates:
        for fn in fields:
            if fn.lower() == cand.lower():
                return fn
    # Substring match next
    for cand in candidates:
        for fn in fields:
            if cand in fn.lower():
                return fn
    return None


def verify_and_calibrate(
    rec: dict,
    parsed: ParsedDocument,
    fields: Dict[str, dict],
    config: Optional[Any] = None,
) -> Dict[str, dict]:
    """Verifies grounding and calibrates confidence scores for extracted fields.

    Args:
        rec: Raw record containing schema, tags, doc_id, etc.
        parsed: ParsedDocument containing raw_text, nodes, get_node_with_neighbors, etc.
        fields: Dict of extracted fields, where each field is:
                {"value": ..., "confidence": float, "evidence_span": ..., "quote": ...}
        config: Optional configuration dict or VerifierConfig instance.

    Returns:
        Dict of verified and calibrated fields with updated confidence and metadata.
    """
    cfg = config if isinstance(config, VerifierConfig) else VerifierConfig(
        risk_json_mult=_get_cfg(config, "risk_json_mult", 0.90),
        risk_corrupted_mult=_get_cfg(config, "risk_corrupted_mult", 0.85),
        grounding_fuzzy_threshold=_get_cfg(config, "grounding_fuzzy_threshold", 75.0),
        cuad_span_fuzzy_threshold=_get_cfg(config, "cuad_span_fuzzy_threshold", 80.0),
        ungrounded_penalty=_get_cfg(config, "ungrounded_penalty", 0.20),
        financial_boost=_get_cfg(config, "financial_boost", 0.03),
        financial_penalty=_get_cfg(config, "financial_penalty", 0.50),
        financial_tol=_get_cfg(config, "financial_tol", 0.02),
        financial_max_boost=_get_cfg(config, "financial_max_boost", 0.995),
    )

    schema_props = rec.get("schema", {}).get("properties", {})
    doc_id = rec.get("doc_id", "")
    tags = rec.get("tags") or []
    is_corrupted = (
        parsed.doc_type == "scanned_pdf"
        or "_corrupted" in doc_id
        or "corrupted" in tags
    )
    raw_text_clean = (
        len(parsed.raw_text.strip()) >= 20
        and not ("_corrupted" in doc_id or "corrupted" in tags)
    )

    calibrated_fields: Dict[str, dict] = {}
    field_reasons: Dict[str, List[str]] = {}

    # Stage 1: Individual Field Grounding & Calibration
    for field_name, field_data in fields.items():
        field_out = dict(field_data)
        val = field_out.get("value")
        raw_conf = float(field_out.get("confidence", 0.0) if field_out.get("confidence") is not None else 0.0)
        evidence_span = field_out.get("evidence_span")
        quote = field_out.get("quote")
        prop = schema_props.get(field_name, {})
        kind = prop.get("x-normalizer", "string")
        prop_type = prop.get("type", "")

        reasons: List[str] = []
        conf = raw_conf

        # 1. Null / Empty handling
        if val is None or val == "":
            field_out["value"] = None
            field_out["confidence"] = 0.0
            calibrated_fields[field_name] = field_out
            field_reasons[field_name] = ["null_or_empty"]
            if log_verification_detail:
                log_verification_detail(
                    field=field_name,
                    value=None,
                    raw_conf=raw_conf,
                    calibrated_conf=0.0,
                    reason="null_or_empty",
                    evidence_span=evidence_span,
                    logger=logger,
                )
            continue

        # 2. Checkbox Exception
        # If value in ("false", False, "true", True) or property is boolean:
        # do NOT penalize value in ("false", False) for lacking a quote or raw-text match
        is_bool_val = (
            isinstance(val, bool)
            or str(val).lower() in ("true", "false")
        )
        is_bool_prop = (prop_type == "boolean" or kind == "boolean")
        is_boolean = is_bool_val or is_bool_prop
        is_unchecked_checkbox = is_boolean and (
            val in (False, "false", "False")
            or (isinstance(val, str) and val.strip().lower() == "false")
        )

        val_str = str(val).strip()
        quote_str = str(quote).strip() if quote is not None else ""

        # 3. Node Neighbor Grounding
        node_grounded = False
        node_text = ""
        if evidence_span:
            span_str = str(evidence_span)
            if hasattr(parsed, "get_node_with_neighbors") and callable(getattr(parsed, "get_node_with_neighbors")):
                node_text = parsed.get_node_with_neighbors(span_str) or ""
            elif hasattr(parsed, "get_node_text") and callable(getattr(parsed, "get_node_text")):
                node_text = parsed.get_node_text(span_str) or ""
            elif hasattr(parsed, "nodes") and isinstance(parsed.nodes, dict):
                node_text = parsed.nodes.get(span_str) or parsed.nodes.get(span_str.strip("[]")) or ""

        if node_text:
            node_lower = node_text.lower()
            val_lower = val_str.lower()
            quote_lower = quote_str.lower()

            # Exact match
            if val_lower and val_lower in node_lower:
                node_grounded = True
                reasons.append("node_exact_match")
            elif quote_lower and quote_lower in node_lower:
                node_grounded = True
                reasons.append("node_quote_exact_match")
            else:
                # Normalized match
                norm_match = False
                try:
                    norm_v = normalize(val, kind)
                    if norm_v:
                        if kind == "amount":
                            norm_nv = norm_amount(node_text)
                            if norm_v in node_text or (norm_nv and norm_v in norm_nv):
                                norm_match = True
                        elif kind == "date":
                            norm_nv = norm_date(node_text)
                            if norm_v in node_text or (norm_nv and norm_v in norm_nv):
                                norm_match = True
                        elif kind == "id":
                            norm_nv = norm_id(node_text)
                            if norm_v in norm_nv:
                                norm_match = True
                        else:
                            norm_nv = normalize(node_text, kind)
                            if norm_v in norm_nv:
                                norm_match = True
                except Exception:
                    pass

                if norm_match:
                    node_grounded = True
                    reasons.append("node_normalized_match")
                else:
                    # Fuzzy match
                    r_val = fuzz.partial_ratio(val_lower, node_lower) if val_lower else 0.0
                    r_quote = fuzz.partial_ratio(quote_lower, node_lower) if quote_lower else 0.0
                    if r_val >= cfg.grounding_fuzzy_threshold or r_quote >= cfg.grounding_fuzzy_threshold:
                        node_grounded = True
                        reasons.append(f"node_fuzzy_match(val={r_val:.1f},quote={r_quote:.1f})")

        # 4. Document Raw Text Grounding Check
        doc_grounded = False
        if is_unchecked_checkbox:
            # Checkbox Exception: do NOT penalize value in ("false", False)
            doc_grounded = True
            reasons.append("checkbox_unchecked_exempt")
        elif kind == "span":
            # For CUAD span fields (x-normalizer == "span"):
            # If value contains " | ", split by " | " and verify each segment aligns with parsed.raw_text
            raw_lower = parsed.raw_text.lower()
            if " | " in val_str:
                segments = [s.strip() for s in val_str.split(" | ") if s.strip()]
                if not segments:
                    doc_grounded = False
                else:
                    seg_matches: List[bool] = []
                    for seg in segments:
                        seg_l = seg.lower()
                        if seg_l in raw_lower:
                            seg_matches.append(True)
                        else:
                            ratio = fuzz.partial_ratio(seg_l, raw_lower)
                            seg_matches.append(ratio >= cfg.cuad_span_fuzzy_threshold)

                    if all(seg_matches):
                        doc_grounded = True
                        reasons.append(f"cuad_all_spans_grounded({len(segments)})")
                    elif any(seg_matches):
                        doc_grounded = True
                        # Partially aligned multi-span: scale confidence proportionally
                        match_ratio = sum(seg_matches) / len(segments)
                        conf *= max(0.40, match_ratio)
                        reasons.append(f"cuad_partial_spans_grounded({sum(seg_matches)}/{len(segments)})")
                    else:
                        doc_grounded = False
                        reasons.append("cuad_no_spans_grounded")
            else:
                # Single span
                val_l = val_str.lower()
                if val_l in raw_lower:
                    doc_grounded = True
                    reasons.append("cuad_single_span_exact")
                else:
                    ratio = fuzz.partial_ratio(val_l, raw_lower)
                    if ratio >= cfg.cuad_span_fuzzy_threshold:
                        doc_grounded = True
                        reasons.append(f"cuad_single_span_fuzzy({ratio:.1f})")
                    else:
                        doc_grounded = False
                        reasons.append(f"cuad_single_span_ungrounded({ratio:.1f})")
        else:
            # Non-span fields
            raw_lower = parsed.raw_text.lower()
            val_lower = val_str.lower()
            quote_lower = quote_str.lower()

            if val_lower and val_lower in raw_lower:
                doc_grounded = True
                reasons.append("doc_exact_match")
            elif quote_lower and quote_lower in raw_lower:
                doc_grounded = True
                reasons.append("doc_quote_exact_match")
            elif node_grounded:
                # Grounded in indexed node tree
                doc_grounded = True
                reasons.append("doc_node_tree_match")
            elif is_boolean and any(w in raw_lower for w in re.split(r"[\W_]+", field_name.lower()) if len(w) > 2):
                # Boolean field name exists on page
                doc_grounded = True
                reasons.append("checkbox_field_present")
            else:
                # Normalized doc check
                norm_doc_match = False
                try:
                    norm_v = normalize(val, kind)
                    if norm_v:
                        if kind == "amount":
                            norm_doc = norm_amount(parsed.raw_text)
                            if norm_v in parsed.raw_text or (norm_doc and norm_v in norm_doc):
                                norm_doc_match = True
                        elif kind == "date":
                            norm_doc = norm_date(parsed.raw_text)
                            if norm_v in parsed.raw_text or (norm_doc and norm_v in norm_doc):
                                norm_doc_match = True
                        elif kind == "id":
                            norm_doc = norm_id(parsed.raw_text)
                            if norm_v in norm_doc:
                                norm_doc_match = True
                except Exception:
                    pass

                if norm_doc_match:
                    doc_grounded = True
                    reasons.append("doc_normalized_match")
                else:
                    # Fuzzy match
                    r_val = fuzz.partial_ratio(val_lower, raw_lower) if val_lower else 0.0
                    r_quote = fuzz.partial_ratio(quote_lower, raw_lower) if quote_lower else 0.0
                    if r_val >= cfg.grounding_fuzzy_threshold or r_quote >= cfg.grounding_fuzzy_threshold:
                        doc_grounded = True
                        reasons.append(f"doc_fuzzy_match(val={r_val:.1f},quote={r_quote:.1f})")

        # 5. Penalize completely ungrounded values
        if not doc_grounded and raw_text_clean:
            conf *= cfg.ungrounded_penalty  # Drop conf *= 0.20
            reasons.append("penalty_ungrounded_in_raw_text")

        # 6. Check node localization consistency
        if evidence_span:
            if node_grounded:
                reasons.append("node_location_verified")
            elif doc_grounded:
                # Value exists in document, but cited node did not contain it
                conf *= 0.90
                reasons.append("penalty_node_span_mismatch")

        field_out["confidence"] = conf
        calibrated_fields[field_name] = field_out
        field_reasons[field_name] = reasons

    # Stage 2: Financial Arithmetic Gate
    subtotal_fn = _match_field_name(calibrated_fields, SUBTOTAL_CANDIDATES)
    total_fn = _match_field_name(calibrated_fields, TOTAL_CANDIDATES)

    if subtotal_fn and total_fn:
        sub_val = _parse_amount(calibrated_fields[subtotal_fn].get("value"))
        tot_val = _parse_amount(calibrated_fields[total_fn].get("value"))

        if sub_val is not None and tot_val is not None:
            tax_fn = _match_field_name(calibrated_fields, TAX_CANDIDATES)
            disc_fn = _match_field_name(calibrated_fields, DISCOUNT_CANDIDATES)

            tax_val = 0.0
            has_tax = False
            if tax_fn:
                tv = _parse_amount(calibrated_fields[tax_fn].get("value"))
                if tv is not None:
                    tax_val = tv
                    has_tax = True

            disc_val = 0.0
            has_disc = False
            if disc_fn:
                dv = _parse_amount(calibrated_fields[disc_fn].get("value"))
                if dv is not None:
                    disc_val = dv
                    has_disc = True

            expected = sub_val + tax_val - disc_val
            diff = abs(expected - tot_val)

            participating_fields = [subtotal_fn, total_fn]
            if has_tax and tax_fn:
                participating_fields.append(tax_fn)
            if has_disc and disc_fn:
                participating_fields.append(disc_fn)

            if diff <= cfg.financial_tol:
                # Arithmetic balances! Boost confidence on participating fields
                for pfn in participating_fields:
                    cur_conf = calibrated_fields[pfn]["confidence"]
                    calibrated_fields[pfn]["confidence"] = min(cfg.financial_max_boost, cur_conf + cfg.financial_boost)
                    field_reasons[pfn].append(f"fin_gate_pass(diff={diff:.2f})")

                if log_financial_gate:
                    log_financial_gate(
                        status="PASS",
                        expected=expected,
                        actual=tot_val,
                        diff=diff,
                        fields_affected=participating_fields,
                        logger=logger,
                    )
            else:
                # Arithmetic clearly disagrees! Penalize confidence on participating fields
                for pfn in participating_fields:
                    calibrated_fields[pfn]["confidence"] *= cfg.financial_penalty
                    field_reasons[pfn].append(f"fin_gate_fail(diff={diff:.2f})")

                if log_financial_gate:
                    log_financial_gate(
                        status="FAIL",
                        expected=expected,
                        actual=tot_val,
                        diff=diff,
                        fields_affected=participating_fields,
                        logger=logger,
                    )

    # Stage 3: Configurable Risk Multipliers & Final Clamping
    for field_name, field_out in calibrated_fields.items():
        val = field_out.get("value")
        if val is None or val == "":
            field_out["value"] = None
            field_out["confidence"] = 0.0
            continue

        conf = field_out["confidence"]
        prop = schema_props.get(field_name, {})
        kind = prop.get("x-normalizer", "string")

        # JSON risk multiplier
        if kind == "json":
            conf *= cfg.risk_json_mult
            field_reasons[field_name].append(f"risk_json_mult({cfg.risk_json_mult:.2f})")

        # Corrupted / Scanned document risk multiplier
        if is_corrupted:
            conf *= cfg.risk_corrupted_mult
            field_reasons[field_name].append(f"risk_corrupted_mult({cfg.risk_corrupted_mult:.2f})")

        # Clamp between 0.0 and 1.0, round to 4 decimal places
        conf = max(0.0, min(1.0, conf))
        field_out["confidence"] = round(conf, 4)

        if log_verification_detail:
            raw_c = float(fields[field_name].get("confidence", 0.0) if fields[field_name].get("confidence") is not None else 0.0)
            log_verification_detail(
                field=field_name,
                value=field_out.get("value"),
                raw_conf=raw_c,
                calibrated_conf=field_out["confidence"],
                reason="; ".join(field_reasons.get(field_name, [])),
                evidence_span=field_out.get("evidence_span"),
                logger=logger,
            )

    return calibrated_fields
