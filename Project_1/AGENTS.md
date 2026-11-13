# Agent Instructions & Operational Rules

## 1. Zero Tolerance for Cheating / Shortcuts

Any agent working on this project MUST produce **generalizable, real solutions**. The submission will be evaluated on a **hidden test set** (`run_hidden.sh <input.jsonl> <output.jsonl>`) where test documents do not have labels.

### STRICT RULES:

1. **NEVER read or parse `qas` / `answers` from `data/raw/cuad/CUADv1.json`:**
   - `CUADv1.json` happens to contain training labels from the Atticus Project benchmark.
   - On the hidden test set, `qas` will be empty, stripped, or absent.
   - Any code that peeks into `qa["answers"]` or `qa["is_impossible"]` is disqualified data leakage.
   - You may ONLY read the contract text (`context` or `paragraphs.context`).
2. **NEVER memorize fixed values from `dev.jsonl`:**
   - Do NOT emit hardcoded template values because 100% of a dev batch shared an address or company name.
   - Values MUST be extracted from the target document.
3. **Generalization is the ONLY goal:**
   - If an approach would output `null` or garbage when given an unseen contract or an unseen tax form, it is invalid.

---

## 2. LLM Infrastructure: SIXT LiteLLM Endpoint

SIXT provides a LiteLLM proxy endpoint with an API key.

- **Base URL:** `https://llm.orange.sixt.com/v1` (or `https://llm.orange.sixt.com/`)
- **API Key:** Located in `.env` as `SIXT_LLM_KEY` (or `LLM_API_KEY`)
- **Compatibility:** Standard OpenAI-compatible API (`/chat/completions`, `/models`) or LiteLLM client.
- When calling models, use the standard OpenAI wire format (`messages`, `model`, `temperature`, etc.).

---

## 3. Real Engineering Requirements

1. **Document Ingestion & Retrieval:**
   - **CUAD:** Contracts can be 100+ pages long. Do NOT truncate at 60k chars blindly. Use paragraph chunking, BM25, or semantic clause retrieval to locate the relevant contract sections, then extract verbatim spans.
   - **ExtractBench:** Scans, PDFs, forms, checkboxes. Use text extraction (`pdftotext` / `pypdf`) or multimodal vision models for scanned/handwritten documents.
2. **True Verbatim Grounding Guardrail:**
   - Any extracted string or span from an LLM must be cross-checked against the raw source text.
   - If the LLM generates a paraphrased or non-existent value, it must be rejected or zeroed out (`confidence = 0.0, value = null`).
3. **Calibrated Confidence Scoring:**
   - The business cost is 1 for human review vs 100 for a wrong accepted value.
   - Confidence must reflect real probability of correctness: only assign $\ge 0.95$ when grounding, type formatting, and extraction consistency pass all checks.
   - If in doubt, emit `null` with `confidence = 0.0`. Saying "I don't know" is the core metric.
