# Task: Upgrade `stage1_pipeline/` for >99% Precision, Adaptive VLM/Text Routing, Consensus, and Side-by-Side Benchmarking

## Current State & Strict Rules

We already have `stage1_pipeline/` (`node_tree.py`, `parser.py`, `extractor.py`, `test_pipeline.py`) working[cite: 38]. Keep our existing `Node`, `ParsedDocument`, and `[sec-X/p-Y/s-Z]` canonical path architecture intact[cite: 38].

- **NEVER** modify `2_prepare.py`, `5_score.py`, `test_score.py`, or `stage1_pipeline/test_pipeline.py`[cite: 1, 38].
- **ZERO CHEATING:** Never inspect `qas` or `answers` in `CUADv1.json` (keep `load_cuad_corpus()` strictly as-is)[cite: 38].
- **Cost Asymmetry:** A wrong or hallucinated accepted value costs **100** vs. **1** for abstaining (`null` or `< threshold`)[cite: 1]. Accepted fields must exceed **99% precision**[cite: 1].
- Never run an LLM config on full `data/unified/dev.jsonl` until we verify it on `data/unified/dev_mini.jsonl` (30 docs).

---

## Upgrades to Build on `stage1_pipeline/`

### 1. Extend `stage1_pipeline/parser.py` with Page Image Rendering & Neighbor Node Lookup

- Add a helper `render_pdf_pages_base64(pdf_path: Path, max_pages: int = 10, dpi: int = 150) -> List[str]` using `pymupdf` (`fitz`) (or `pdf2image` fallback) so short/scanned PDFs and images can be passed directly to multimodal VLMs.
- Add `get_node_with_neighbors(self, path: str) -> str` on `ParsedDocument` that returns the text of `path` plus its immediate previous and next sibling nodes (in case an extracted span crosses a sentence or table-row boundary)[cite: 38].

### 2. Add Adaptive Routing & Schema Chunking (`stage1_pipeline/router.py`)

Create `router.py` to decide how each `(rec, parsed)` pair is processed based on config thresholds:

- **Adaptive Modality Routing (`route_mode: "auto" | "vlm" | "text"`):**
  - **Use VLM (Page Images + Schema):** When `parsed.doc_type in ("scanned_pdf", "image")` OR (`parsed.doc_type == "digital_pdf"` AND `parsed.page_count <= config.vlm_max_pages` [default: 5]). Optionally attach `parsed.layout_text` alongside the images if `parsed.doc_type == "digital_pdf"`.
  - **Use Text Model (Location-Annotated Node Tree Markdown):** When `parsed.doc_type == "cuad"` OR `parsed.page_count > config.vlm_max_pages` (long documents)[cite: 10, 38].
- **Full-Context / Chunked Document Handling (Remove 90k Truncation):**
  - Remove the hard `MAX_CHARS_PROMPT = 90_000` cutoff in `extractor.py`[cite: 38]. If a CUAD contract exceeds `config.max_doc_chars` (e.g., 180,000 chars), split `parsed.layout_text` along section boundaries with overlap, extract per chunk, and merge (preferring non-null predictions with higher confidence).
- **Schema Chunking for Dense Forms:**
  - If `len(schema["properties"]) > config.schema_chunk_size` (default: 20), split the schema properties into chunks of `schema_chunk_size` fields, extract each chunk concurrently, and merge the resulting `fields` dictionary. (ExtractBench forms with 150+ fields fail or truncate without schema chunking[cite: 31, 32].)

### 3. Upgrade `stage1_pipeline/extractor.py` (Async Concurrency + Better Prompting + Smart Grounding)

- **Async Concurrency & Retries:** Add async batch extraction (`asyncio.Semaphore(config.concurrency)`, default 10) with exponential backoff retries on rate limits/timeouts, while preserving the existing synchronous `extract_document()` and JSONL resume checkpointing[cite: 38].
- **Improve `EXTRACTION_PROMPT`:**
  - Ask for `{"quote": "<exact verbatim snippet from doc or null>", "value": ..., "confidence": ..., "evidence_span": "<canonical_path_tag or null>"}`.
  - Add explicit rules:
    1. **Atomic Extraction:** Never merge adjacent fields (e.g., keep town, county, and state separate)[cite: 37].
    2. **Dotted-Leader Forms:** Include both the row label and value in `"quote"` to prevent vertical row-shifting[cite: 36, 37].
    3. **CUAD Multi-Spans (`x-normalizer == "span"`):** Copy verbatim text; join multiple non-contiguous clauses with `" | "`[cite: 1].
    4. **Checkboxes (`boolean`):** Return `"true"` if marked, `"false"` if clearly unchecked on a visible page, `null` if the page/section is absent[cite: 1, 22].
- **Fix Grounding & Calibration (`stage1_pipeline/verifiers.py`):**
  - Import and reuse `5_score.py`'s exact normalizers (`norm_string`, `norm_id`, `norm_amount`, `norm_date`, `norm_json`, `span_f1`, `values_match`)[cite: 1].
  - **Checkbox Exception:** Do NOT penalize `value in ("false", False)` for lacking a quote or raw-text match (unchecked boxes have no mark to cite)[cite: 22].
  - **Normalized & Fuzzy Grounding (Replace brittle `is_grounded` zeroing):**
    - Check if `quote` or `value` matches `parsed.get_node_with_neighbors(evidence_span)` using `rapidfuzz.fuzz.partial_ratio >= 75` (or normalized match via `norm_amount`/`norm_date`/`norm_id`)[cite: 1, 38].
    - For CUAD `span` fields, split `value` by `" | "` and verify each segment aligns with `parsed.raw_text` (`partial_ratio >= 80`)[cite: 1, 38]. If completely ungrounded in `parsed.raw_text` (and text was extracted cleanly), drop confidence (`conf *= 0.20`) rather than silently mutating `value` to `None`, so threshold sweeps in `5_score.py` remain observable[cite: 1].
  - **Financial Arithmetic Gate:** When `subtotal`, `vat_amount`/`tax`, `discount_amount`, and `total` are present, check `abs((subtotal + tax - discount) - total) <= 0.02`. Boost confidence (`min(0.995, conf + 0.03)`) if balanced; penalize (`conf *= 0.50`) if inconsistent.
  - **Configurable Risk Multipliers:** Apply config multipliers for high-risk slices, e.g., `json` fields (`x-normalizer == "json"`, scored all-or-nothing[cite: 1]) and `scanned_pdf` / `_corrupted` docs[cite: 1].

### 4. Multi-Method Strategies & Side-by-Side Benchmark Runner (`configs/` & `benchmark.py`)

Create JSON configs in `configs/` so we can run and compare methods side-by-side via `python3 benchmark.py --data data/unified/dev_mini.jsonl --configs ...`:

1. `configs/m1_routed_single.json`: Adaptive VLM (short/scanned) + Node-Tree Text (long/CUAD) + Schema Chunking + Verifiers.
2. `configs/m2_dual_consensus.json`:
   - **Pass 1:** Primary routed model (`m1`).
   - **Pass 2 (Non-Null Fast-Path):** Only for fields where Pass 1 returned a **non-null** value, run a second pass using either a **second distinct model** OR the **complementary modality** (e.g., if Pass 1 used VLM images on a digital PDF, Pass 2 uses the Node-Tree text model; if Pass 1 used Text Model A on CUAD, Pass 2 uses Text Model B).
   - **Agreement Rule:** Compare Pass 1 and Pass 2 with `5_score.py`'s `values_match(v1, v2, kind)`[cite: 1]. If they agree, boost confidence (`max(c1, c2, 0.97)`); if they disagree, drop confidence (`min(c1, c2) * 0.40`) so a human reviews it[cite: 1].
3. `configs/m3_critic_verified.json`: `m1` followed by a fast Adversarial Critic prompt on non-null fields verifying whether the value is unambiguously supported without guessing.

---

## Immediate Execution Steps

1. Verify `pymupdf` (`fitz`) and `rapidfuzz` are installed in `.venv` (install if missing).
2. Create `data/unified/dev_mini.jsonl` with 30 stratified documents (15 `cuad` + 15 `extractbench`, including `_corrupted` and multi-page PDFs)[cite: 1].
3. Implement the upgrades above while keeping `python3 stage1_pipeline/test_pipeline.py` passing[cite: 38].
4. Run `python3 benchmark.py --data data/unified/dev_mini.jsonl --configs configs/m1_routed_single.json configs/m2_dual_consensus.json` and print the comparison table (`precision`, `coverage @ precision >= 0.99`, `hallucination_rate`, `best_threshold`, `net_saving_pct`, `runtime_sec`) plus the top 10 false positives (`wrong` or `hallucinated`)[cite: 1].
