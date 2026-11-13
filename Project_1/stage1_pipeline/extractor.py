"""Stage 1 Extractor with Provider Routing (LiteLLM & OpenAI) and Node-Tree Grounding.

Uses layout-aware parsed documents from parser.py, requests canonical evidence_span paths
([sec-1/p-2/s-5]), and validates verbatim grounding against the node subtree and raw text.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root and package directory are in sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv

try:
    from stage1_pipeline.parser import ParsedDocument, parse_document
except ImportError:
    from parser import ParsedDocument, parse_document

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_SIXT_URL = "https://llm.orange.sixt.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_CHARS_PROMPT = 90_000

EXTRACTION_PROMPT = """You are an expert, precision-first information extraction model.
Extract the fields below from the document. Reply with ONE valid JSON object and nothing else.

Output format:
{{
  "<field_name>": {{
    "value": <string, number, boolean, JSON array/object, or null>,
    "confidence": <float 0.0 to 1.0>,
    "evidence_span": "<canonical_path_tag or null>"
  }},
  ...
}}

Rules:
1. Copy values EXACTLY as printed in the document. Do not summarize, guess, or calculate.
2. If a field is not present or cannot be read with certainty, return "value": null, "confidence": 0.0, "evidence_span": null.
   CRITICAL: A wrong accepted value costs 100 times more than a null. If unsure, output null.
3. "evidence_span" must be ONLY the canonical location tag of the node containing the value, e.g. "[sec-1/p-2/s-5]" or "[sec-2/table-0/row-3]".
   Do NOT quote text inside evidence_span, only output the bracketed location path tag string.
4. "confidence" is your calibrated certainty that the extracted value is 100% correct and verbatim.

Fields to extract:
{fields}

--- DOCUMENT (with Canonical Path Tags) ---
{document}
"""


def describe_schema(schema: dict) -> str:
    """Formats schema properties into concise field descriptions."""
    out = []
    properties = schema.get("properties", {})
    for name, spec in properties.items():
        desc = spec.get("description") or spec.get("title") or spec.get("type") or ""
        kind = spec.get("x-normalizer")
        kind_str = f" [{kind}]" if kind else ""
        desc_str = f": {str(desc)[:250]}" if desc else ""
        out.append(f"- {name}{kind_str}{desc_str}")
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
    """Verifies that an extracted value is grounded verbatim in source text."""
    if text is None or value is None or isinstance(value, bool):
        return True
    if not isinstance(value, (str, int, float)):
        return True  # Nested structures (e.g. lists/dicts)
    val_str = str(value).strip()
    if not val_str or len(val_str) > 200:
        return True
    squash = lambda s: re.sub(r"[\s,\.\-_/]+", "", str(s).lower())
    return squash(val_str) in squash(text)


class Extractor:
    """Unified Extractor supporting LiteLLM and OpenAI provider backends."""

    def __init__(
        self,
        provider: str = "litellm",
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = (
            api_key
            or os.environ.get("SIXT_LLM_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        base = (
            api_base
            or os.environ.get("SIXT_LLM_URL")
            or os.environ.get("LLM_BASE_URL")
            or DEFAULT_SIXT_URL
        )
        # Ensure base URL format
        base = base.rstrip("/")
        if not base.endswith("/v1") and "azure" not in base:
            base = base + "/v1"
        self.api_base = base

        self.model = (
            model
            or os.environ.get("SIXT_LLM_MODEL")
            or os.environ.get("LLM_MODEL")
            or DEFAULT_MODEL
        )

        self._openai_client = None
        if self.provider == "openai":
            from openai import OpenAI
            self._openai_client = OpenAI(base_url=self.api_base, api_key=self.api_key or "sk-dummy")

    def call_llm(self, prompt: str) -> str:
        """Invokes LLM using configured provider (litellm or openai)."""
        if not self.api_key:
            raise ValueError("No API key configured for LLM. Set SIXT_LLM_KEY or LLM_API_KEY.")

        messages = [{"role": "user", "content": prompt}]

        if self.provider == "litellm":
            os.environ["LITELLM_TELEMETRY"] = "False"
            import litellm
            litellm.telemetry = False

            litellm_model = self.model
            if not any(litellm_model.startswith(p) for p in ("openai/", "azure/", "bedrock/", "vertex_ai/")):
                litellm_model = f"openai/{litellm_model}"

            resp = litellm.completion(
                model=litellm_model,
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0,
                timeout=120,
            )
            return resp.choices[0].message.content

        elif self.provider == "openai":
            if self._openai_client is None:
                from openai import OpenAI
                self._openai_client = OpenAI(base_url=self.api_base, api_key=self.api_key)

            resp = self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                timeout=120,
            )
            return resp.choices[0].message.content

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def extract_from_parsed(self, rec: dict, parsed: ParsedDocument) -> dict:
        """Extracts fields from a ParsedDocument and applies node grounding."""
        schema = rec.get("schema", {})
        doc_text = parsed.layout_text[:MAX_CHARS_PROMPT]
        prompt = EXTRACTION_PROMPT.format(
            fields=describe_schema(schema),
            document=doc_text,
        )

        raw_reply = self.call_llm(prompt)
        parsed_reply = parse_json_reply(raw_reply)

        fields: Dict[str, Any] = {}
        properties = schema.get("properties", {})

        for field_name in properties:
            item = parsed_reply.get(field_name)
            if not isinstance(item, dict):
                item = {"value": item}

            val = item.get("value")
            conf = item.get("confidence", 0.0)
            evidence_span = item.get("evidence_span")

            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
            conf = min(max(conf, 0.0), 1.0)

            # Grounding and Calibration Guardrail
            if val is not None:
                # 1. Check Node Subtree Grounding
                node_grounded = False
                if evidence_span:
                    node_text = parsed.get_node_text(str(evidence_span))
                    if node_text and is_grounded(val, node_text):
                        node_grounded = True

                # 2. Check Document Raw Text Grounding
                doc_grounded = is_grounded(val, parsed.raw_text)

                if not doc_grounded:
                    # Hallucination detected! Strictly penalize or abstain
                    logger.info("Un-grounded value for %s: %r. Zeroing confidence.", field_name, val)
                    conf = 0.0
                    val = None
                elif node_grounded:
                    # Verified at exact canonical node location
                    conf = min(conf, 0.99)
                else:
                    # Present in document but not in specified node
                    conf = min(conf * 0.85, 0.90)

            fields[field_name] = {
                "value": val,
                "confidence": round(conf, 3),
                "evidence_span": evidence_span,
            }

        return {"doc_id": rec["doc_id"], "fields": fields}


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


def extract_document(rec: dict, extractor: Optional[Extractor] = None, mode: str = "auto") -> dict:
    """End-to-end stage separated extraction for one record:
    1. Parse with parser.py -> ParsedDocument
    2. Extract with extractor.py -> result dict
    """
    parsed = parse_document(rec)

    has_api_key = bool(os.environ.get("SIXT_LLM_KEY") or os.environ.get("LLM_API_KEY"))
    use_llm = mode == "llm" or (mode == "auto" and has_api_key)

    if use_llm:
        ext = extractor or Extractor()
        return ext.extract_from_parsed(rec, parsed)
    else:
        return offline_extract_from_parsed(rec, parsed)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Layout-Aware Document Extractor")
    parser.add_argument("--data", required=True, help="Input dataset JSONL (e.g. data/unified/dev.jsonl)")
    parser.add_argument("--out", required=True, help="Output prediction JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Process first N records")
    parser.add_argument("--provider", choices=["litellm", "openai"], default="litellm", help="LLM backend provider")
    parser.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto", help="Execution mode")
    parser.add_argument("--model", default=None, help="LLM model identifier")
    parser.add_argument("--api-base", default=None, help="LLM API base URL")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    args = parser.parse_args()

    extractor = None
    if args.mode != "offline":
        extractor = Extractor(
            provider=args.provider,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
        )

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

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, rec in enumerate(recs):
            doc_id = rec["doc_id"]
            if doc_id in done_ids:
                continue

            try:
                result = extract_document(rec, extractor=extractor, mode=args.mode)
            except Exception as e:
                logger.exception("Error extracting %s: %s", doc_id, e)
                result = {
                    "doc_id": doc_id,
                    "fields": {f: {"value": None, "confidence": 0.0} for f in rec.get("schema", {}).get("properties", {})},
                }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i+1}/{len(recs)}] Done: {doc_id}", file=sys.stderr)


if __name__ == "__main__":
    main()

