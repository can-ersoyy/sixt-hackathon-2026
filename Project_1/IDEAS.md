# Project 1: Ideas & Strategies for "Teach a Machine to Say 'I Don't Know'"

> **Core Objective:** Maximize **Net Saving** under an asymmetric cost model:
>
> - Human review cost = **1**
> - Wrong booked value / hallucination cost = **100**
> - **Breakeven precision threshold is > 99%.** An error rate over 1% will yield negative savings (worse than doing nothing).
> - Abstaining (`null` or confidence < threshold) is safe (costs 1). A confident false positive is catastrophic (costs 100).

---

## 1. Grounding & Anti-Hallucination Mechanisms

Because wrong answers cost 100× more than abstaining, preventing hallucinations and verifying grounding is the single most critical factor.

### 1.1 Strict Verbatim Substring Matching

- **Concept:** For any extracted string or span, check whether it exists verbatim (or normalized: whitespace collapsed, lowercase, punctuation-insensitive) in the document source text.
- **Action:** If the extracted text cannot be found in the document text, **immediately drop confidence to 0.0 or set value to `null`**.
- **Impact:** Eliminates generative paraphrasing hallucinations (e.g., an LLM explaining a clause in its own words instead of extracting the exact contract text).

### 1.2 "Absence-First" Two-Stage Extraction

- **Stage 1 (Binary Existence Classifier):** First ask: *"Is field X explicitly present and clearly stated in this document? Answer YES or NO."*
- **Stage 2 (Extraction):** Only if Stage 1 is an unequivocal "YES" with high certainty, proceed to extract the exact text snippet.
- **Why it matters:** In CUAD, 80–90% of contract clauses are absent (`null`). Forcing an extractor to answer often induces generative models to fabricate plausible-sounding clauses.

### 1.3 Verbatim Quote / Evidence Requester

- Require the model to output two fields for every schema item:
  1. `quote`: The exact, continuous snippet from the document proving the value.
  2. `value`: The target extracted value.
- If `quote` is not found verbatim in the document text, discard `value`.

---

## 2. Confidence Calibration & Self-Consistency

LLM raw self-reported confidence numbers are notoriously overconfident (models will report 0.95 confidence on hallucinations). We need empirical calibration.

### 2.1 Multi-Sample Self-Consistency (Sample & Agree)

- Run extraction 2 or 3 times (e.g., temperature 0.2 or slightly varied prompts / orderings).
- If outputs from multiple runs match (using normalized string match or high token F1), assign **high confidence** (e.g., 0.98).
- If outputs disagree or any run returns `null`, assign **low confidence** (e.g., 0.50) or collapse to `null`.
- This is one of the most reliable signals of true certainty.

### 2.2 LLM Critic / Verification Pass (Second Opinion)

- Pass the extracted value and the document context to a separate verification prompt:
  > *"Review this document snippet. Is field '[Field Name]' unequivocally equal to '[Extracted Value]' without ambiguity? If there is any doubt, respond REJECT."*
  >
- Penalize confidence heavily on any hesitation or rejection.

### 2.3 Post-Hoc Confidence Penalty Checklist

Construct a deterministic confidence score formula based on observable risk factors:

$$
C_{\text{final}} = C_{\text{base}} \times \prod \text{penalties}
$$

- **Length Penalty:** Very long extracted spans have higher risk of boundary errors or extraneous text.
- **OCR Quality Penalty:** If text has unusual character entropy or high proportion of unrecognized symbols, discount confidence.
- **Ambiguity Penalty:** If multiple candidate numbers/dates appear in the same section, penalize.
- **Format Match Boost/Penalty:** If a date, EIN, SSN, or percentage strictly matches expected regex patterns, maintain confidence; if it violates standard formatting, slash confidence.

---

## 3. Dataset-Specific Architectures

The problem contains two fundamentally different document domains: **CUAD** (long legal contracts) and **ExtractBench** (messy scans, tax forms, receipts, checkboxes).

### 3.1 CUAD (Contract Legal Spans)

- **Problem:** Contracts exceed token limits (starter script cuts off at 60,000 chars, losing everything past page 15-20).
- **Chunking & Retrieval (BM25 / Semantic Search):**
  - Index the contract by paragraphs/sections.
  - For each of the 41 CUAD clause types (e.g., "Governing Law", "Audit Rights", "Non-Compete"), retrieve the top 3-5 most relevant paragraphs using BM25 or keyword queries.
  - Run the extractor specifically on the retrieved paragraphs. This avoids context truncation and sharpens attention.
- **Span Extraction vs Normalization:**
  - CUAD evaluation uses **token F1 $\ge 0.8$**.
  - Extract the exact sentence/paragraph boundary rather than attempting to summarize.
- **Clause Prior Filter:**
  - Some clauses are almost universal ("Governing Law", "Agreement Date"), while others are extremely rare. Use dev set priors to adjust confidence baselines.

### 3.2 ExtractBench (Forms, Scans, Checkboxes, Handwriting)

- **Multimodal (Vision) Routing:**
  - When `pdf_text` fails, returns empty, or yields garbled layout, switch to vision mode (render PDF pages as images and send to a multimodal LLM).
- **Checkbox Handling Strategy:**
  - Checkboxes (e.g. Schedule K-1 "Final K-1", "Amended K-1") are binary (`true`/`false`).
  - OCR often fails on checkboxes. An unmarked box is `false`, a marked box is `true`.
  - Because false positives are lethal, if the visual signal is ambiguous (smudge, low res), fall back to `false` (if baseline is overwhelmingly false) or keep confidence below threshold to defer to human review.
- **Tabular & Coded Array Parsing:**
  - In fields like `box_14_self_employment_earnings` or coded box entries, the expected output is a JSON array of `{"code": ..., "amount": ...}`.
  - Parse with strict JSON schema validation. If JSON is malformed, drop confidence.

---

## 4. Normalization Alignment (Free Precision Boosts)

Matching `5_score.py`'s exact normalization rules directly prevents accidental mismatches:

- **Amounts (`norm_amount`):** Automatically strip currency symbols (`$`, `€`), handle European decimal notation (`1.234,50` $\to$ `1234.50`), and convert percentages (`20%` $\to$ `0.20`).
- **Dates (`norm_date`):** Standardize all extracted dates into ISO `YYYY-MM-DD` matching `DATE_FORMATS` in `5_score.py`.
- **IDs (`norm_id`):** Strip spaces, hyphens, and non-alphanumeric characters where appropriate.
- **Strings (`norm_string`):** Unicode normalize (`NFKC`), collapse extra whitespace, casefold.

---

## 5. Threshold & Policy Optimization

- **Global vs Per-Dataset Threshold:**
  - Tune separate confidence cutoffs for CUAD vs ExtractBench if permitted, or calibrate raw confidence so that a single global threshold (e.g., `0.95` or `0.98`) works optimally across both.
- **Conservative Operating Point:**
  - In a 100:1 loss regime, it is far better to accept only 20% of fields at 99.8% precision (yielding +19% net saving) than 70% of fields at 96% precision (yielding a disastrous -210% loss).
- **Threshold Grid Search on Dev:**
  - Generate a full PR curve and net savings curve on `data/unified/dev.jsonl` across thresholds `[0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]`.
  - Pick the threshold with safety margin: choose a threshold slightly higher than the empirical peak to guard against distribution shift on the hidden test set.

---

## 6. Hybrid & Tiered Execution Pipeline

To balance latency, API cost, and accuracy:

```
Document Input
  │
  ├─► Is it CUAD?
  │     ├─► Chunk & BM25 / Keyword Clause Retrieval
  │     ├─► Targeted Span Extraction
  │     └─► Verbatim Substring Check & F1 Verification
  │
  ├─► Is it ExtractBench?
  │     ├─► Has clean PDF text?
  │     │     ├─ Yes: Text-based Schema Extractor + Regex Format Validation
  │     │     └─ No / Scan: Vision API (multimodal image prompt)
  │     └─► Normalizer (Amount, Date, Checkbox)
  │
  ▼
Cross-Validation & Confidence Assignment
  │
  ├─► Substring Grounding Verified?  (No ──► conf = 0.0 / null)
  ├─► Self-Consistency Disagreement? (Yes ──► conf = 0.40)
  ├─► Type / Regex Match Valid?      (No ──► conf = 0.20)
  └─► Model Confidence Calibrated
  │
  ▼
Threshold Gate (e.g. >= 0.95)
  ├─► Accept & Automate (Net +1 saving per right field)
  └─► Defer to Human (Safe fallback, cost = 1)
```

---

## 7. Architectural Pattern: Hierarchical Node Tree & Dual-LLM Consensus

### 7.1 Hierarchical Node Tree Representation
Documents are parsed (via Docling for layout PDFs/scans, or structured paragraph splitting for CUAD) into a typed node tree:
- `SectionNode`: `[sec-X]`
- `ParagraphNode`: `[sec-X/p-Y]`
- `SentenceNode`: `[sec-X/p-Y/s-Z]`
- `TableNode`: `[sec-X/table-Y]`
- `TableRowNode`: `[sec-X/table-Y/row-Z]`
- `ImageNode`: `[page-X/img-Y]`

### 7.2 Location-Only `evidence_span` Output Schema
To minimize token cost and latency, the LLMs return only the canonical address path as evidence:
```json
{
  "<field_name>": {
    "value": "...",
    "confidence": 0.98,
    "evidence_span": "[sec-1/p-2/s-5]"
  }
}
```
- Emits ~5 tokens per field instead of quoting 50-100 words of legal text.
- Allows the verifier to load `node_tree["[sec-1/p-2/s-5]"]` directly for isolated grounding verification (< 100 tokens per evaluation instead of 50k tokens).

### 7.3 Dual-Model Cross-Validation
- Query two diverse models (e.g. `gpt-4o-mini` and `gemini-3.5-flash`).
- Agreement on `value` AND `evidence_span` node path $\to$ Confidence $\ge 0.95$.
- Disagreement or divergence in cited nodes $\to$ Confidence dropped below threshold / `null` (safe human review, cost = 1).

---

## 8. Fast Experimentation Roadmap

1. **Phase 1: Exploratory Data Analysis (`EDA.md`)**
   - Measure class balance, null frequencies, and field types in `dev.jsonl`.
2. **Phase 2: Stage 1 Node Tree Parser (`stage1_pipeline/parser.py`)**
   - Layout-aware Docling parser with canonical `[sec-X/p-Y/s-Z]` paths.
3. **Phase 3: Stage 2 Dual-LLM Extractor (`stage1_pipeline/extractor.py`)**
   - CLI `--provider [openai|litellm]` returning `value`, `confidence`, and location-only `evidence_span`.
4. **Phase 4: Consensus & Grounding Verifier**
   - Cross-model agreement and isolated node substring checks.
5. **Phase 5: Final Threshold Sweep & `run_hidden.sh` Hardening**
   - Lock in optimal threshold, run full evaluation, and verify execution runtime.
