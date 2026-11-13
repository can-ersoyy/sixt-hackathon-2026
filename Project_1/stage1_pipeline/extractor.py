"""Stage 1 Extractor with Adaptive Provider Routing, Async Concurrency,
Multimodal VLM Support, Document/Schema Chunking, and Calibrated Verification.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root and package directory are in sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv

try:
    from stage1_pipeline.parser import ParsedDocument, parse_document
    from stage1_pipeline.router import (
        PipelineConfig,
        RouteDecision,
        determine_route,
        chunk_document_text,
        merge_chunk_predictions,
        split_schema,
        merge_schema_predictions,
    )
    from stage1_pipeline.verifiers import verify_and_calibrate
    from stage1_pipeline.logging_utils import (
        get_logger,
        log_document_start,
        log_route_decision,
        log_llm_request,
        log_llm_response,
        log_verification_detail,
        log_progress,
        log_stage_banner,
    )
except ImportError:
    from parser import ParsedDocument, parse_document
    from router import (
        PipelineConfig,
        RouteDecision,
        determine_route,
        chunk_document_text,
        merge_chunk_predictions,
        split_schema,
        merge_schema_predictions,
    )
    from verifiers import verify_and_calibrate
    from logging_utils import (
        get_logger,
        log_document_start,
        log_route_decision,
        log_llm_request,
        log_llm_response,
        log_verification_detail,
        log_progress,
        log_stage_banner,
    )

load_dotenv()
logger = get_logger("stage1_pipeline.extractor")

DEFAULT_SIXT_URL = "https://llm.orange.sixt.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_CHARS_PROMPT = 90_000

CUAD_SHORT_SPAN_FIELDS = {
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
}

EXTRACTION_PROMPT = """You are an expert, precision-first information extraction model.
Extract the fields below from the document. Reply with ONE valid JSON object and nothing else.

Output format:
{{
  "<field_name>": {{
    "quote": "<exact verbatim snippet from document or null>",
    "value": <string, number, boolean, JSON array/object, or null>,
    "confidence": <float 0.0 to 1.0>
  }},
  ...
}}

Strict Extraction Rules:
1. ATOMIC EXTRACTION: Follow each field's exact description and formatting rules. Never merge adjacent fields.
2. DOTTED-LEADER FORMS: Include both row label and value in "quote" to prevent vertical row-shifting.
3. CUAD SHORT SPANS ([short_span]): For Document Name, Parties, Agreement Date, Effective Date, extract ONLY the exact title, party names/aliases (separated by " | "), or date string as printed — NOT the full preamble sentence.
4. CUAD CLAUSE SPANS ([clause_span]): Copy the COMPLETE verbatim sentence(s) or contractual clause from the document (do NOT extract a short fragment like 'State of New York' or 'one year' alone, and NEVER return true/false for a [clause_span] field). If multiple clauses apply, join them with " | ".
5. CHECKBOXES (boolean): Return true if marked, false if clearly unchecked on a visible page, null if the page or section is absent (unless description specifies TWO-STATE).
6. ABSOLUTE VERBATIM GROUNDING: Copy values EXACTLY as printed in the document. Do not summarize, paraphrase, guess, or calculate.
7. ABSTENTION POLICY (CRITICAL): If a field is not explicitly present or cannot be identified with high certainty, return "value": null, "confidence": 0.0, "quote": null. NEVER return "None", "N/A", "Not applicable", or "None (...)" as a string — use JSON null.
   COST ASYMMETRY: A wrong accepted value costs 100x more than a null. Only assign confidence >= 0.95 when 100% certain, unambiguous, and verbatim. Use 0.70-0.85 if partially uncertain.
8. NEGATED / DISCLAIMED CLAUSES: Do NOT extract clauses that explicitly negate or disclaim the requested right/restriction (e.g., 'no third-party beneficiaries', 'not intended to confer upon any person... rights', 'as-is / no warranties', 'non-exclusive' when asked for Exclusivity). If the contract states the right/restriction does NOT exist, return null.

Fields to extract:
{fields}

--- DOCUMENT CONTENT ---
{document}
"""

CRITIC_PROMPT = """You are an adversarial information extraction auditor.
A primary extraction model extracted the candidate values below from the document.
Your task is to determine whether each candidate value is unambiguously, verbatim supported by the document without guessing or hallucination.

Document:
{document}

Candidate Extractions to audit:
{candidates}

For each candidate, respond with:
{{
  "<field_name>": {{
    "supported": true | false,
    "verbatim_quote": "<exact quote from document or null>",
    "auditor_confidence": <float 0.0 to 1.0>,
    "critique": "<short 1-line reason if disputed or supported>"
  }}
}}
Return ONLY ONE valid JSON object.
"""


def _format_type_hint(spec: dict, defs: dict, depth: int = 0) -> str:
    """Formats JSON schema type/structure/enum/items concisely for LLM prompts."""
    if not isinstance(spec, dict) or depth > 2:
        return ""
    if "$ref" in spec:
        ref_name = str(spec["$ref"]).split("/")[-1]
        if ref_name in defs:
            return _format_type_hint(defs[ref_name], defs, depth + 1)
        return ref_name

    if "anyOf" in spec and isinstance(spec["anyOf"], list):
        parts = [_format_type_hint(sub, defs, depth) for sub in spec["anyOf"]]
        parts = [p for p in parts if p and p != "null"]
        return "|".join(parts) if parts else ""

    t = spec.get("type")
    if isinstance(t, list):
        t = "|".join(x for x in t if x != "null")

    if "enum" in spec and isinstance(spec["enum"], list):
        vals = ", ".join(repr(v) for v in spec["enum"][:10])
        return f"enum({vals})"

    if t == "array" and isinstance(spec.get("items"), dict):
        item_hint = _format_type_hint(spec["items"], defs, depth + 1)
        return f"array<{item_hint}>" if item_hint else "array"

    if t == "object" and isinstance(spec.get("properties"), dict):
        req = set(spec.get("required", []))
        fields_str = []
        for k, v in list(spec["properties"].items())[:25]:
            sub_t = _format_type_hint(v, defs, depth + 1) or v.get("type") or "any"
            star = "*" if k in req else ""
            fields_str.append(f"{k}{star}: {sub_t}")
        return "{" + ", ".join(fields_str) + "}"

    return str(t) if t else ""


def describe_schema(schema: dict) -> str:
    """Formats schema properties into detailed, unambiguous field descriptions."""
    out = []
    properties = schema.get("properties", {})
    defs = schema.get("$defs", {}) if isinstance(schema.get("$defs"), dict) else {}

    for name, spec in properties.items():
        if not isinstance(spec, dict):
            out.append(f"- {name}")
            continue
        desc = spec.get("description") or spec.get("title") or ""
        kind = spec.get("x-normalizer")
        if kind == "span":
            kind_str = " [short_span]" if name in CUAD_SHORT_SPAN_FIELDS else " [clause_span]"
        elif kind:
            kind_str = f" [{kind}]"
        else:
            kind_str = ""

        type_hint = _format_type_hint(spec, defs)
        type_str = f" ({type_hint})" if type_hint and type_hint != kind else ""
        desc_str = f": {str(desc)[:1500]}" if desc else ""
        out.append(f"- {name}{kind_str}{type_str}{desc_str}")
    return "\n".join(out)


def parse_json_reply(reply: str) -> dict:
    """Robustly extracts JSON object from LLM response text."""
    s = reply.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    try:
        start = s.index("{")
        end = s.rindex("}") + 1
        return json.loads(s[start:end])
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse JSON reply: %s", reply[:200])
        return {}


def is_grounded(value: Any, text: Optional[str]) -> bool:
    """Verifies that an extracted value is grounded verbatim in source text.
    Maintained for backward compatibility.
    """
    if text is None or value is None or isinstance(value, bool):
        return True
    if not isinstance(value, (str, int, float)):
        return True
    val_str = str(value).strip()
    if not val_str or len(val_str) > 200:
        return True
    squash = lambda s: re.sub(r"[\s,\.\-_/]+", "", str(s).lower())
    return squash(val_str) in squash(text)


def resolve_llm_config(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Resolves LLM credentials and endpoints from arguments or environment."""
    resolved_key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("SIXT_LLM_KEY")
        or ""
    )
    resolved_base = (
        api_base
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("SIXT_LLM_URL")
        or DEFAULT_SIXT_URL
    ).rstrip("/")

    # Normalize OpenAI-compatible endpoint base URL
    if not (
        resolved_base.endswith("/v1")
        or resolved_base.endswith("openai")
        or "azure" in resolved_base
        or "googleapis.com" in resolved_base
    ):
        resolved_base = resolved_base + "/v1"

    resolved_model = (
        model
        or os.environ.get("LLM_MODEL")
        or os.environ.get("SIXT_LLM_MODEL")
        or DEFAULT_MODEL
    )
    return resolved_key, resolved_base, resolved_model


class Extractor:
    """Synchronous / Unified Extractor supporting LiteLLM and OpenAI backends."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.provider = provider.lower()
        self.config = config or PipelineConfig()
        cfg_model = getattr(self.config, "model_text", None) if self.config else None
        target_model = model or cfg_model
        # If config has hardcoded default but LLM_MODEL is explicitly set in env, prefer env
        if os.environ.get("LLM_MODEL") and target_model in ("gpt-4o-mini", None):
            target_model = os.environ.get("LLM_MODEL")

        self.api_key, self.api_base, self.model = resolve_llm_config(api_key, api_base, target_model)
        self._openai_client = None
        if self.api_key:
            from openai import OpenAI
            self._openai_client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def call_llm(
        self,
        prompt: str,
        images_b64: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Synchronously invokes LLM with text or multimodal messages."""
        if not self.api_key:
            raise ValueError("No API key configured for LLM. Set SIXT_LLM_KEY or LLM_API_KEY.")

        chosen_model = model or self.model
        if log_llm_request is not None:
            log_llm_request(model=chosen_model, prompt_chars=len(prompt), logger=logger)

        if images_b64:
            content_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            messages = [{"role": "user", "content": content_items}]
        else:
            messages = [{"role": "user", "content": prompt}]

        last_err = None
        for attempt in range(1, max_retries + 1):
            t0 = time.perf_counter()
            try:
                if self.provider == "litellm":
                    os.environ["LITELLM_TELEMETRY"] = "False"
                    import litellm
                    litellm.telemetry = False
                    l_model = chosen_model
                    if not any(l_model.startswith(p) for p in ("openai/", "azure/", "bedrock/", "vertex_ai/")):
                        l_model = f"openai/{l_model}"
                    resp = litellm.completion(
                        model=l_model,
                        messages=messages,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        temperature=0,
                        timeout=120,
                    )
                    latency = time.perf_counter() - t0
                    tokens = getattr(resp, "usage", None)
                    token_dict = {
                        "prompt_tokens": getattr(tokens, "prompt_tokens", 0) if tokens else 0,
                        "completion_tokens": getattr(tokens, "completion_tokens", 0) if tokens else 0,
                        "total_tokens": getattr(tokens, "total_tokens", 0) if tokens else 0,
                    }
                    if log_llm_response is not None:
                        log_llm_response(model=chosen_model, latency_sec=latency, status="200 OK", tokens=token_dict, logger=logger)
                    return resp.choices[0].message.content
                else:
                    if self._openai_client is None:
                        from openai import OpenAI
                        self._openai_client = OpenAI(base_url=self.api_base, api_key=self.api_key)
                    resp = self._openai_client.chat.completions.create(
                        model=chosen_model,
                        messages=messages,
                        temperature=0,
                        timeout=120,
                    )
                    latency = time.perf_counter() - t0
                    tokens = getattr(resp, "usage", None)
                    token_dict = {
                        "prompt_tokens": getattr(tokens, "prompt_tokens", 0) if tokens else 0,
                        "completion_tokens": getattr(tokens, "completion_tokens", 0) if tokens else 0,
                        "total_tokens": getattr(tokens, "total_tokens", 0) if tokens else 0,
                    }
                    if log_llm_response is not None:
                        log_llm_response(model=chosen_model, latency_sec=latency, status="200 OK", tokens=token_dict, logger=logger)
                    return resp.choices[0].message.content or ""
            except Exception as e:
                latency = time.perf_counter() - t0
                last_err = e
                logger.warning("LLM attempt %d/%d failed: %s (latency: %.2fs)", attempt, max_retries, e, latency)
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)

        if log_llm_response is not None:
            log_llm_response(model=chosen_model, latency_sec=0.0, status=f"Failed after {max_retries} attempts: {last_err}", logger=logger)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")

    def extract_from_parsed(self, rec: dict, parsed: ParsedDocument) -> dict:
        """Synchronously extracts fields using pipeline routing, schema chunking & calibration."""
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # In an active event loop, run extraction synchronously
            return self._extract_sync(rec, parsed)
        else:
            async_ext = AsyncExtractor(
                provider=self.provider,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                config=self.config,
            )
            return loop.run_until_complete(async_ext.extract_record_async(rec, parsed))

    def _extract_sync(self, rec: dict, parsed: ParsedDocument) -> dict:
        """Direct sync path when called inside an already-running event loop."""
        route = determine_route(rec, parsed, self.config)
        schema = rec.get("schema", {})
        sub_schemas = split_schema(schema, chunk_size=self.config.schema_chunk_size)
        sub_preds: List[Dict[str, dict]] = []

        for sub_s in sub_schemas:
            fields_desc = describe_schema(sub_s)
            if route.modality == "vlm":
                doc_text = parsed.raw_text if parsed.doc_type == "digital_pdf" else (route.companion_text or "")
                prompt = EXTRACTION_PROMPT.format(fields=fields_desc, document=doc_text)
                reply = self.call_llm(prompt, images_b64=route.images_b64, model=route.model)
                sub_preds.append(parse_json_reply(reply))
            else:
                clean_text = parsed.raw_text or parsed.layout_text
                chunks = chunk_document_text(
                    clean_text,
                    max_chars=self.config.max_doc_chars,
                    overlap_chars=self.config.chunk_overlap_chars,
                )
                if len(chunks) > 1:
                    chunk_preds = []
                    for chunk in chunks:
                        prompt = EXTRACTION_PROMPT.format(fields=fields_desc, document=chunk)
                        reply = self.call_llm(prompt, model=route.model)
                        chunk_preds.append(parse_json_reply(reply))
                    sub_preds.append(merge_chunk_predictions(chunk_preds))
                else:
                    prompt = EXTRACTION_PROMPT.format(fields=fields_desc, document=chunks[0])
                    reply = self.call_llm(prompt, model=route.model)
                    sub_preds.append(parse_json_reply(reply))

        merged_raw = merge_schema_predictions(sub_preds)
        calibrated = verify_and_calibrate(rec, parsed, merged_raw, self.config)
        return {"doc_id": rec["doc_id"], "fields": calibrated}


class AsyncExtractor:
    """Asynchronous Extractor with concurrency control, retries, and multimodal routing."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.provider = provider.lower()
        self.config = config or PipelineConfig()
        cfg_model = getattr(self.config, "model_text", None) if self.config else None
        target_model = model or cfg_model
        if os.environ.get("LLM_MODEL") and target_model in ("gpt-4o-mini", None):
            target_model = os.environ.get("LLM_MODEL")

        self.api_key, self.api_base, self.model = resolve_llm_config(api_key, api_base, target_model)
        self.semaphore = asyncio.Semaphore(self.config.concurrency)

        from openai import AsyncOpenAI
        self._async_client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key or "sk-dummy")

    async def call_llm_async(
        self,
        prompt: str,
        images_b64: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Asynchronously invokes LLM with concurrency gating and exponential backoff."""
        if not self.api_key:
            raise ValueError("No API key configured for LLM. Set SIXT_LLM_KEY or LLM_API_KEY.")

        chosen_model = model or self.model
        if log_llm_request is not None:
            log_llm_request(model=chosen_model, prompt_chars=len(prompt), logger=logger)

        if images_b64:
            content_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            messages = [{"role": "user", "content": content_items}]
        else:
            messages = [{"role": "user", "content": prompt}]

        last_err = None
        async with self.semaphore:
            for attempt in range(1, max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await self._async_client.chat.completions.create(
                        model=chosen_model,
                        messages=messages,
                        temperature=0,
                        timeout=120,
                    )
                    latency = time.perf_counter() - t0
                    tokens = getattr(resp, "usage", None)
                    token_dict = {
                        "prompt_tokens": getattr(tokens, "prompt_tokens", 0) if tokens else 0,
                        "completion_tokens": getattr(tokens, "completion_tokens", 0) if tokens else 0,
                        "total_tokens": getattr(tokens, "total_tokens", 0) if tokens else 0,
                    }
                    if log_llm_response is not None:
                        log_llm_response(model=chosen_model, latency_sec=latency, status="200 OK", tokens=token_dict, logger=logger)
                    return resp.choices[0].message.content or ""
                except Exception as e:
                    latency = time.perf_counter() - t0
                    last_err = e
                    logger.warning("Async LLM attempt %d/%d failed: %s (latency: %.2fs)", attempt, max_retries, e, latency)
                    if attempt < max_retries:
                        await asyncio.sleep(1.5 * attempt)

        if log_llm_response is not None:
            log_llm_response(model=chosen_model, latency_sec=0.0, status=f"Failed after {max_retries} attempts: {last_err}", logger=logger)
        raise RuntimeError(f"Async LLM call failed after {max_retries} retries: {last_err}")

    async def _extract_subschema(
        self,
        sub_s: dict,
        route: RouteDecision,
        parsed: ParsedDocument,
    ) -> Dict[str, dict]:
        """Extracts fields for one sub-schema, handling multimodal or document chunks."""
        fields_desc = describe_schema(sub_s)

        if route.modality == "vlm":
            doc_text = parsed.raw_text if parsed.doc_type == "digital_pdf" else (route.companion_text or "")
            prompt = EXTRACTION_PROMPT.format(fields=fields_desc, document=doc_text)
            reply = await self.call_llm_async(prompt, images_b64=route.images_b64, model=route.model)
            return parse_json_reply(reply)

        clean_text = parsed.raw_text or parsed.layout_text
        chunks = chunk_document_text(
            clean_text,
            max_chars=self.config.max_doc_chars,
            overlap_chars=self.config.chunk_overlap_chars,
        )
        if len(chunks) > 1:
            chunk_tasks = []
            for chunk in chunks:
                prompt = EXTRACTION_PROMPT.format(fields=fields_desc, document=chunk)
                chunk_tasks.append(self.call_llm_async(prompt, model=route.model))
            replies = await asyncio.gather(*chunk_tasks)
            chunk_preds = [parse_json_reply(r) for r in replies]
            return merge_chunk_predictions(chunk_preds)
        else:
            prompt = EXTRACTION_PROMPT.format(fields=fields_desc, document=chunks[0])
            reply = await self.call_llm_async(prompt, model=route.model)
            return parse_json_reply(reply)

    async def extract_record_async(self, rec: dict, parsed: ParsedDocument) -> dict:
        """Executes full extraction pipeline for one record asynchronously."""
        route = determine_route(rec, parsed, self.config)
        schema = rec.get("schema", {})
        sub_schemas = split_schema(schema, chunk_size=self.config.schema_chunk_size)

        tasks = [self._extract_subschema(sub_s, route, parsed) for sub_s in sub_schemas]
        sub_preds = await asyncio.gather(*tasks)

        merged_raw = merge_schema_predictions(sub_preds)
        calibrated = verify_and_calibrate(rec, parsed, merged_raw, self.config)
        return {"doc_id": rec["doc_id"], "fields": calibrated}


def offline_extract_from_parsed(rec: dict, parsed: ParsedDocument) -> dict:
    """Offline keyword and regex rule extraction fallback."""
    PATTERNS = {
        "amount": r"-?\d[\d.,]*\d|\d",
        "date": r"\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|[A-Z][a-z]{2,8}\.? \d{1,2},? \d{4}|\d{1,2} [A-Z][a-z]{2,8}\.? \d{4}",
        "id": r"[A-Za-z0-9][A-Za-z0-9\-/]{3,}",
    }

    text = parsed.raw_text
    fields: Dict[str, Any] = {}
    properties = rec.get("schema", {}).get("properties", {})

    for name, spec in properties.items():
        kind = spec.get("x-normalizer")
        words = [w for w in re.split(r"[\W_]+", name.lower()) if len(w) > 2]
        value, conf, span = None, 0.0, None

        if text and words:
            if kind == "span":
                for path, n_text in parsed.nodes.items():
                    if all(w in n_text.lower() for w in words) and 20 < len(n_text) < 600:
                        value = " ".join(n_text.split())
                        conf = 0.35
                        span = path
                        break
            elif kind in PATTERNS:
                for path, n_text in parsed.nodes.items():
                    low = n_text.lower()
                    if all(w in low for w in words):
                        tail = n_text[low.rindex(words[-1]) + len(words[-1]):]
                        m = re.search(PATTERNS[kind], tail)
                        if m:
                            value = m.group(0).strip()
                            conf = 0.50
                            span = path
                            break

        fields[name] = {"value": value, "confidence": conf, "evidence_span": span}

    return {"doc_id": rec["doc_id"], "fields": fields}


def extract_document(
    rec: dict,
    extractor: Optional[Extractor] = None,
    mode: str = "auto",
    config: Optional[PipelineConfig] = None,
) -> dict:
    """End-to-end stage separated extraction for one record:
    1. Parse with parser.py -> ParsedDocument
    2. Extract with extractor.py -> result dict
    """
    parsed = parse_document(rec)

    has_api_key = bool(os.environ.get("SIXT_LLM_KEY") or os.environ.get("LLM_API_KEY"))
    use_llm = mode == "llm" or (mode == "auto" and has_api_key)

    if use_llm:
        ext = extractor or Extractor(config=config)
        return ext.extract_from_parsed(rec, parsed)
    else:
        return offline_extract_from_parsed(rec, parsed)


async def extract_dataset_async(
    recs: List[dict],
    out_path: Path,
    config: PipelineConfig,
    provider: str = "openai",
    model: Optional[str] = None,
    done_ids: Optional[set] = None,
) -> List[dict]:
    """Extracts a dataset of records concurrently with async checkpointing."""
    if done_ids is None:
        done_ids = set()

    extractor = AsyncExtractor(
        provider=provider,
        model=model or config.model_text,
        config=config,
    )

    results: List[dict] = []
    t_start = time.perf_counter()
    total = len(recs)
    processed = 0

    async def _process_one(idx: int, rec: dict) -> dict:
        nonlocal processed
        doc_id = rec["doc_id"]
        if doc_id in done_ids:
            return {"doc_id": doc_id, "skipped": True}

        try:
            parsed = parse_document(rec)
            res = await extractor.extract_record_async(rec, parsed)
        except Exception as e:
            logger.exception("Async extraction error on %s: %s", doc_id, e)
            res = {
                "doc_id": doc_id,
                "fields": {
                    f: {"value": None, "confidence": 0.0, "evidence_span": None}
                    for f in rec.get("schema", {}).get("properties", {})
                },
            }

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()

        processed += 1
        elapsed = time.perf_counter() - t_start
        if log_progress is not None:
            log_progress(current=processed, total=total, elapsed_sec=elapsed, doc_id=doc_id, logger=logger)
        return res

    tasks = [_process_one(i, rec) for i, rec in enumerate(recs)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if not r.get("skipped")]


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Layout-Aware Document Extractor")
    parser.add_argument("--data", required=True, help="Input dataset JSONL (e.g. data/unified/dev.jsonl)")
    parser.add_argument("--out", required=True, help="Output prediction JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Process first N records")
    parser.add_argument("--provider", choices=["litellm", "openai"], default="openai", help="LLM backend provider")
    parser.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto", help="Execution mode")
    parser.add_argument("--model", default=None, help="LLM model identifier")
    parser.add_argument("--api-base", default=None, help="LLM API base URL")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--concurrency", type=int, default=10, help="Async concurrency limit")
    args = parser.parse_args()

    cfg = PipelineConfig(concurrency=args.concurrency)
    out_path = Path(args.out)
    done_ids = set()
    if out_path.exists():
        done_ids = {
            json.loads(line)["doc_id"]
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    print(f"Loaded {len(recs)} records. Mode: {args.mode}, Provider: {args.provider}", file=sys.stderr)

    if args.mode == "offline":
        with open(out_path, "a", encoding="utf-8") as fout:
            for i, rec in enumerate(recs):
                if rec["doc_id"] in done_ids:
                    continue
                result = extract_document(rec, mode="offline")
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
    else:
        asyncio.run(
            extract_dataset_async(
                recs=recs,
                out_path=out_path,
                config=cfg,
                provider=args.provider,
                model=args.model,
                done_ids=done_ids,
            )
        )


if __name__ == "__main__":
    main()
