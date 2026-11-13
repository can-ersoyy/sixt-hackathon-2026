#!/usr/bin/env python3
"""Side-by-side Benchmark Runner comparing extraction strategies on unified dev/test sets.

Usage:
    python3 benchmark.py --data data/unified/dev_mini.jsonl \
        --configs configs/m1_routed_single.json configs/m2_dual_consensus.json \
        [--limit 30] [--out-dir runs/]
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root in sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv

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
from stage1_pipeline.extractor import AsyncExtractor, parse_json_reply, describe_schema, EXTRACTION_PROMPT, CRITIC_PROMPT
from stage1_pipeline.logging_utils import (
    get_logger,
    setup_logger,
    log_document_start,
    log_route_decision,
    log_llm_request,
    log_llm_response,
    log_verification_detail,
    log_consensus_comparison,
    log_progress,
    log_stage_banner,
)

load_dotenv()
logger = get_logger("benchmark")

# Dynamic import of 5_score normalizers & evaluation logic
_SCORE_MOD = importlib.import_module("5_score")
values_match = _SCORE_MOD.values_match
collect = _SCORE_MOD.collect
risk_coverage = _SCORE_MOD.risk_coverage
headline = _SCORE_MOD.headline
at_threshold = _SCORE_MOD.at_threshold


# =====================================================================
# Strategy Executors
# =====================================================================

async def run_m1_strategy(
    rec: dict,
    parsed: ParsedDocument,
    config: PipelineConfig,
    extractor: AsyncExtractor,
) -> dict:
    """Strategy M1: Routed Single Pass (VLM vs Text) + Schema Chunking + Verifiers."""
    return await extractor.extract_record_async(rec, parsed)


async def run_m2_strategy(
    rec: dict,
    parsed: ParsedDocument,
    config: PipelineConfig,
    extractor: AsyncExtractor,
    extractor_pass2: AsyncExtractor,
) -> dict:
    """Strategy M2: Dual Fast Consensus.
    
    Pass 1: Routed extraction (M1).
    Pass 2: Re-extracts ONLY non-null fields using a complementary modality or second model.
    Agreement: values_match() -> boost to >=0.97; disagree -> penalty 0.40x.
    """
    res1 = await extractor.extract_record_async(rec, parsed)
    fields1 = res1.get("fields", {})

    # Fast path: filter non-null candidate fields
    non_null_fields = {
        k: v for k, v in fields1.items()
        if v.get("value") is not None
    }
    if not non_null_fields:
        return res1

    schema = rec.get("schema", {})
    sub_schema = {
        "type": "object",
        "properties": {k: schema.get("properties", {}).get(k, {}) for k in non_null_fields},
    }

    # Pass 2 Routing: use complementary modality or second model
    route1 = determine_route(rec, parsed, config)
    model_p2 = config.get("model_pass2") or config.model_text

    cfg_p2_dict = dict(config.to_dict()) if hasattr(config, "to_dict") else {
        "vlm_max_pages": config.vlm_max_pages,
        "max_doc_chars": config.max_doc_chars,
        "chunk_overlap_chars": config.chunk_overlap_chars,
        "schema_chunk_size": config.schema_chunk_size,
        "concurrency": config.concurrency,
        "risk_json_mult": config.get("risk_json_mult", 0.90),
        "risk_corrupted_mult": config.get("risk_corrupted_mult", 0.85),
    }

    # If Pass 1 used VLM on digital PDF, Pass 2 uses Text model on node tree
    if route1.modality == "vlm" and parsed.doc_type == "digital_pdf":
        cfg_p2_dict["route_mode"] = "text"
        cfg_p2_dict["model_text"] = model_p2
    else:
        cfg_p2_dict["route_mode"] = config.route_mode
        cfg_p2_dict["model_text"] = model_p2
        cfg_p2_dict["model_vlm"] = model_p2

    cfg_pass2 = PipelineConfig.from_dict(cfg_p2_dict)
    rec_p2 = {**rec, "schema": sub_schema}
    old_cfg_p2 = extractor_pass2.config
    extractor_pass2.config = cfg_pass2
    try:
        res2 = await extractor_pass2.extract_record_async(rec_p2, parsed)
        preds2 = res2.get("fields", {})
    except Exception as e:
        logger.warning("Pass 2 failed on %s: %s", rec["doc_id"], e)
        preds2 = {}
    finally:
        extractor_pass2.config = old_cfg_p2

    boost_target = config.get("boost_conf", 0.97)
    penalty_mult = config.get("penalty_mult", 0.40)
    risk_json_mult = config.get("risk_json_mult", 0.90)
    risk_corrupted_mult = config.get("risk_corrupted_mult", 0.85)
    doc_id = rec.get("doc_id", "")
    is_corrupted = "_corrupted" in doc_id or "corrupted" in (rec.get("tags") or [])
    is_scanned = parsed.doc_type in ("scanned_pdf", "image")
    raw_norm = " ".join(parsed.raw_text.split()).lower() if parsed.raw_text else ""

    # Compare Pass 1 and Pass 2 with 5_score's values_match
    for fname, f1 in non_null_fields.items():
        v1 = f1.get("value")
        c1 = float(f1.get("confidence", 0.0))
        p2_item = preds2.get(fname) or {}
        v2 = p2_item.get("value") if isinstance(p2_item, dict) else p2_item
        c2 = float(p2_item.get("confidence", 0.0)) if isinstance(p2_item, dict) else 0.0

        kind = schema.get("properties", {}).get(fname, {}).get("x-normalizer", "string")

        is_match = False
        if v2 is not None:
            is_match = values_match(v1, v2, kind)

        if is_match:
            # Respect verifier caps from either pass (e.g. cap_fuzzy_only=0.88, cap_reformatted_string=0.88)
            if c1 < 0.90 or c2 < 0.90:
                new_conf = min(0.92, max(c1, c2))
            else:
                new_conf = min(0.98, max(c1, c2, boost_target))

            # Enforce risk caps that must never be overwritten by consensus boost
            if kind == "json":
                new_conf = min(new_conf, risk_json_mult)
            if is_corrupted:
                new_conf = min(new_conf, risk_corrupted_mult)
            if kind == "span":
                # CUAD span_f1 >= 0.80 boundary/alias sensitivity + semantic false positives
                new_conf = min(new_conf, 0.90)
            if is_scanned and new_conf >= 0.95:
                # Require Docling OCR token-boundary confirmation on scanned docs to clear 0.95
                v1_norm = " ".join(str(v1).split()).lower()
                ocr_confirmed = bool(
                    v1_norm
                    and not isinstance(v1, bool)
                    and re.search(
                        r"(?<![A-Za-z0-9$\-])" + re.escape(v1_norm) + r"(?![A-Za-z0-9\-])",
                        raw_norm,
                    )
                )
                if not ocr_confirmed:
                    new_conf = min(new_conf, 0.92)
        else:
            new_conf = min(c1, c2 if v2 is not None else c1) * penalty_mult
            if penalty_mult == 0.0 or new_conf <= 0.0:
                fields1[fname]["value"] = None

        log_consensus_comparison(
            field=fname,
            val1=v1,
            val2=v2,
            match=is_match,
            old_conf=c1,
            new_conf=new_conf,
            logger=logger,
        )
        fields1[fname]["confidence"] = round(new_conf, 4)

    return {"doc_id": rec["doc_id"], "fields": fields1}


async def run_m3_strategy(
    rec: dict,
    parsed: ParsedDocument,
    config: PipelineConfig,
    extractor: AsyncExtractor,
) -> dict:
    """Strategy M3: Adversarial Critic Verification.
    
    Pass 1: Routed extraction (M1).
    Pass 2: Targeted adversarial audit on non-null candidate fields.
    """
    res1 = await extractor.extract_record_async(rec, parsed)
    fields1 = res1.get("fields", {})

    non_null_fields = {
        k: v for k, v in fields1.items()
        if v.get("value") is not None
    }
    if not non_null_fields:
        return res1

    # Formulate critic audit prompt
    candidates = {
        k: {
            "value": v["value"],
            "quote": v.get("quote"),
            "evidence_span": v.get("evidence_span"),
        }
        for k, v in non_null_fields.items()
    }
    critic_doc = parsed.raw_text[:180000] if parsed.raw_text else parsed.layout_text[:180000]
    prompt = CRITIC_PROMPT.format(
        document=critic_doc,
        candidates=json.dumps(candidates, indent=2),
    )

    critic_model = config.get("critic_model") or config.model_text
    try:
        critic_reply = await extractor.call_llm_async(prompt, model=critic_model)
        audit = parse_json_reply(critic_reply)
    except Exception as e:
        logger.warning("Critic call failed on %s: %s", rec["doc_id"], e)
        audit = {}

    dispute_penalty = config.get("critic_dispute_penalty", 0.35)
    support_boost = config.get("critic_support_boost", 0.02)

    for fname, f1 in non_null_fields.items():
        c1 = float(f1.get("confidence", 0.0))
        aud = audit.get(fname, {})
        supported = aud.get("supported", False) if isinstance(aud, dict) else False

        if supported:
            new_conf = min(0.995, c1 + support_boost)
        else:
            new_conf = c1 * dispute_penalty
            if dispute_penalty == 0.0 or new_conf <= 0.0:
                fields1[fname]["value"] = None

        fields1[fname]["confidence"] = round(new_conf, 4)

    return {"doc_id": rec["doc_id"], "fields": fields1}


# =====================================================================
# Strategy Runner & Evaluator
# =====================================================================

async def evaluate_config(
    cfg_path: Path,
    recs: List[dict],
    parsed_docs: Dict[str, ParsedDocument],
    out_dir: Path,
    force: bool = False,
) -> Tuple[Dict[str, Any], List[dict], Dict[str, dict]]:
    """Runs a single configuration against the records and computes evaluation metrics."""
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = PipelineConfig.from_dict(cfg_data)
    name = cfg_data.get("name", cfg_path.stem)

    logger.info("==================================================")
    logger.info("RUNNING BENCHMARK: %s", name)
    logger.info("Description: %s", cfg_data.get("description", ""))
    logger.info("==================================================")

    out_file = out_dir / f"{name}_preds.jsonl"
    done_preds: Dict[str, dict] = {}
    if force and out_file.exists():
        out_file.unlink()
    elif out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                done_preds[item["doc_id"]] = item.get("fields", {})

    extractor = AsyncExtractor(config=cfg, model=cfg.model_text)
    extractor_p2 = AsyncExtractor(
        config=cfg,
        model=cfg.get("model_pass2") or cfg.model_text,
    )

    t0 = time.perf_counter()
    new_preds = {}

    async def _process_doc(rec: dict) -> dict:
        doc_id = rec["doc_id"]
        if doc_id in done_preds:
            return {"doc_id": doc_id, "fields": done_preds[doc_id]}

        parsed = parsed_docs[doc_id]
        if "m2" in name or cfg_data.get("strategy") == "m2_dual_consensus":
            res = await run_m2_strategy(rec, parsed, cfg, extractor, extractor_p2)
        elif "m3" in name or cfg_data.get("strategy") == "m3_critic_verified":
            res = await run_m3_strategy(rec, parsed, cfg, extractor)
        else:
            res = await run_m1_strategy(rec, parsed, cfg, extractor)

        # Write to JSONL checkpoint immediately
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
        return res

    tasks = [_process_doc(r) for r in recs]
    doc_results = await asyncio.gather(*tasks)
    runtime_sec = time.perf_counter() - t0

    preds_by_doc = {r["doc_id"]: r["fields"] for r in doc_results}

    # Evaluate using 5_score collect & curve
    rows = collect(recs, preds_by_doc)
    curve = risk_coverage(rows)
    h99 = headline(curve, target=0.99)
    h95 = headline(curve, target=0.95)

    # Compute Net Saving at optimal threshold
    # Cost model: baseline cost = total_gold * 1; error cost = 100
    total_fields = len(rows)
    best_pt = max(
        curve,
        key=lambda p: (
            p["precision"] >= 0.95,
            (total_fields - (p["correct"] * 0 + (p["wrong"] + p["hallucinated"]) * 100 + (p["missed"] + p["true_null"]) * 1)),
        ),
    ) if curve else at_threshold(rows, 0.95)

    baseline_cost = total_fields * 1
    pred_cost = (
        best_pt["correct"] * 0
        + (best_pt["wrong"] + best_pt["hallucinated"]) * 100
        + (best_pt["missed"] + best_pt["true_null"]) * 1
    )
    net_saving_pct = ((baseline_cost - pred_cost) / baseline_cost) * 100.0 if baseline_cost else 0.0

    # Also calculate metrics at threshold 0.95
    pt_95 = at_threshold(rows, 0.95)

    stats = {
        "name": name,
        "runtime_sec": round(runtime_sec, 2),
        "total_fields": total_fields,
        "precision_at_095": round(pt_95["precision"] * 100, 2),
        "coverage_at_095": round(pt_95["coverage"] * 100, 2),
        "coverage_at_099": round(h99["coverage_at_target_precision"] * 100, 2),
        "reached_099": h99["reached_target"],
        "threshold_at_099": h99["threshold_at_target"],
        "hallucination_rate": round(pt_95["hallucination_rate"] * 100, 3),
        "best_threshold": round(best_pt["threshold"], 4),
        "best_precision": round(best_pt["precision"] * 100, 2),
        "best_coverage": round(best_pt["coverage"] * 100, 2),
        "net_saving_pct": round(net_saving_pct, 2),
        "wrong_count": best_pt["wrong"],
        "hallucinated_count": best_pt["hallucinated"],
    }

    # Extract false positives for diagnostic audit
    false_positives = []
    for r in rows:
        accepted = r["pred_present"] and r["conf"] >= best_pt["threshold"]
        if accepted and not r["match"]:
            doc_id = r["doc_id"]
            field_name = r["field"]
            pred_item = preds_by_doc.get(doc_id, {}).get(field_name, {})
            # Find gold value
            gold_rec = next((x for x in recs if x["doc_id"] == doc_id), None)
            gold_val = gold_rec["gold"].get(field_name) if gold_rec else None

            false_positives.append({
                "doc_id": doc_id,
                "field": field_name,
                "source": r["source"],
                "confidence": r["conf"],
                "pred_val": pred_item.get("value"),
                "gold_val": gold_val,
                "is_hallucination": not r["gold_present"],
            })

    # Sort false positives descending by confidence
    false_positives.sort(key=lambda x: x["confidence"], reverse=True)
    return stats, false_positives, preds_by_doc


def print_comparison_table(all_stats: List[Dict[str, Any]]) -> None:
    """Renders formatted Markdown comparison table."""
    print("\n" + "=" * 110)
    print("SIDE-BY-SIDE BENCHMARK RESULTS")
    print("=" * 110)

    headers = [
        "Config Name",
        "Precision @0.95",
        "Cov @0.95",
        "Cov @>=0.99 Prec",
        "Halluc. Rate",
        "Best Thresh",
        "Net Saving %",
        "Runtime (s)",
    ]
    row_fmt = "| {:<22} | {:>15} | {:>10} | {:>18} | {:>13} | {:>12} | {:>12} | {:>11} |"
    divider = "|:" + "-" * 23 + "|:" + "-" * 16 + "|:" + "-" * 11 + "|:" + "-" * 19 + "|:" + "-" * 14 + "|:" + "-" * 13 + "|:" + "-" * 13 + "|:" + "-" * 12 + "|"

    print(row_fmt.format(*headers))
    print(divider)
    for s in all_stats:
        cov_99_str = f"{s['coverage_at_099']}% (T={s['threshold_at_099']})" if s["reached_099"] else "0.0% (N/A)"
        print(row_fmt.format(
            s["name"],
            f"{s['precision_at_095']}%",
            f"{s['coverage_at_095']}%",
            cov_99_str,
            f"{s['hallucination_rate']}%",
            f"{s['best_threshold']}",
            f"{s['net_saving_pct']}%",
            f"{s['runtime_sec']}s",
        ))
    print("=" * 110 + "\n")


def print_top_false_positives(name: str, fps: List[dict], top_n: int = 10) -> None:
    """Prints top false positives for error diagnosis."""
    print(f"\n--- Top {min(top_n, len(fps))} False Positives for [{name}] ---")
    if not fps:
        print("🎉 ZERO false positives accepted! Perfect precision.")
        return

    for idx, fp in enumerate(fps[:top_n], start=1):
        err_type = "HALLUCINATION (Gold was null)" if fp["is_hallucination"] else "WRONG VALUE"
        print(f"[{idx}] {fp['doc_id']} | Field: '{fp['field']}' ({fp['source']}) | Conf: {fp['confidence']}")
        print(f"    Type: {err_type}")
        print(f"    Pred: {repr(fp['pred_val'])[:120]}")
        print(f"    Gold: {repr(fp['gold_val'])[:120]}")


# =====================================================================
# CLI Entrypoint
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Side-by-Side Benchmark Runner")
    parser.add_argument("--data", default="data/unified/dev_mini.jsonl", help="Dataset path")
    parser.add_argument("--configs", nargs="+", required=True, help="List of JSON config paths")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--out-dir", default="runs", help="Output directory for predictions")
    parser.add_argument("--force", action="store_true", help="Force re-run and overwrite existing checkpoint predictions")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: dataset file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    recs = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    logger.info("Loaded %d documents from %s for benchmarking.", len(recs), data_path)

    # Pre-parse documents once to isolate extraction & model time
    logger.info("Pre-parsing %d documents...", len(recs))
    parsed_docs: Dict[str, ParsedDocument] = {}
    for i, r in enumerate(recs, start=1):
        doc_id = r["doc_id"]
        parsed_docs[doc_id] = parse_document(r)
        if i % 10 == 0 or i == len(recs):
            logger.info("Pre-parsed [%d/%d] docs", i, len(recs))

    all_stats: List[Dict[str, Any]] = []
    fps_by_config: Dict[str, List[dict]] = {}

    for cfg_str in args.configs:
        cfg_file = Path(cfg_str)
        if not cfg_file.exists():
            print(f"Warning: config file {cfg_file} not found, skipping.", file=sys.stderr)
            continue

        stats, fps, _ = asyncio.run(evaluate_config(cfg_file, recs, parsed_docs, out_dir, force=args.force))
        all_stats.append(stats)
        fps_by_config[stats["name"]] = fps

    # Print Side-by-Side Comparison
    print_comparison_table(all_stats)

    # Print Top 10 False Positives for each configuration
    for s in all_stats:
        print_top_false_positives(s["name"], fps_by_config.get(s["name"], []), top_n=10)


if __name__ == "__main__":
    main()

