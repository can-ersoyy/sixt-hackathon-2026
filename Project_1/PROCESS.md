# Project 1: Process & Architecture Log

## Objective
Build a production-grade, generalizable document extraction and calibrated confidence system under the **100:1 asymmetric error penalty regime**:
- Human review cost: **1**
- Wrong accepted value / hallucination cost: **100**
- Required precision for positive ROI: **> 99%**
- Default to abstaining (`null` / confidence = 0.0) whenever ambiguous.

---

## Architecture Decision Record (ADR-001): Two-Stage Node Tree & Dual-LLM Consensus

### 1. Motivation
- **Token Cost & Latency:** Feeding entire 50-page legal contracts (10k–50k+ tokens) repeatedly to LLMs is prohibitively expensive, slow, and prone to "lost-in-the-middle" attention degradation.
- **Asymmetric Penalty:** In a 100:1 error penalty regime, single-model self-reported confidence is unreliable (models report 0.95+ on hallucinations).
- **Grounding Verifiability:** Extracted values must be strictly verifiable against the exact source location in the document without re-parsing or full-document scanning.

### 2. Stage 1: Document Classification & Hierarchical Node Tree Parsing
Before running extraction, documents are classified and converted into a typed **Hierarchical Node Tree**:

```
Document Input
  │
  ├──► Classification:
  │      ├─ CUAD: Long legal contract text (load full context from CUADv1.json context)
  │      └─ ExtractBench: Digital PDF vs Scanned PDF vs Image
  │
  └──► Layout-Aware Parsing (Docling + OCR):
         Constructs a typed Node Tree with canonical hierarchical address paths:
         - SectionNode:      [sec-1], [sec-2], ...
         - ParagraphNode:    [sec-1/p-1], [sec-1/p-2], ...
         - SentenceNode:     [sec-1/p-2/s-1], [sec-1/p-2/s-2], ...
         - TableNode:        [sec-2/table-0], ...
         - TableRowNode:     [sec-2/table-0/row-1], [sec-2/table-0/row-2], ...
         - ImageNode:        [page-1/img-0], ...
```

- Each node stores its text, bbox/page metadata, and child/parent links.
- The document text presented to models embeds these bracketed location tags (e.g. `[sec-7/p-1/s-1]`).

### 3. Stage 2: Dual-LLM Extraction with Location-Only Evidence
- **Supported Providers:** Modular CLI support for `--provider [openai|litellm]` connecting to SIXT LiteLLM endpoint (`https://llm.orange.sixt.com/v1`).
- **Prompt Output Schema:**
  ```json
  {
    "<field_name>": {
      "value": "<extracted string, boolean, number, or null>",
      "confidence": <float 0.0 to 1.0>,
      "evidence_span": "[sec-X/p-Y/s-Z]"
    }
  }
  ```
- **Location-Only Optimization:** `evidence_span` returns **only the canonical path tag** (e.g. `"[sec-1/p-2/s-5]"`), **NOT** the raw text quote. This saves hundreds of generation tokens per query, drastically reducing latency and output token cost.

### 4. Stage 3: Consensus, Subtree Grounding, and Confidence Calibration
```
               Model A (e.g. gpt-4o-mini)          Model B (e.g. gemini-3.5-flash)
                          │                                       │
                          ▼                                       ▼
            {"val": "A", "loc": "[s-1/p-2]"}        {"val": "A", "loc": "[s-1/p-2]"}
                          │                                       │
                          └───────────────────┬───────────────────┘
                                              │
                                              ▼
                                 [1. Consensus Gate]
                     Do both models agree on normalized value & node?
                          ├─ Disagreement / one says null ──► conf = 0.0, val = null (Route to Human)
                          └─ Full Agreement               ──► Proceed to Grounding
                                              │
                                              ▼
                             [2. Isolated Subtree Grounding]
                  Fetch node text: node_tree.get("[s-1/p-2]")
                  Check: Is extracted value strictly grounded in node text?
                          ├─ Not grounded / outside node  ──► conf = 0.0, val = null
                          └─ Verbatim / fuzzy match pass  ──► Proceed
                                              │
                                              ▼
                             [3. Confidence Assignment]
                  - Both agree + strictly grounded + type regex valid: confidence = 0.98
                  - Any doubt / ambiguity / contradiction:             confidence = 0.0 / null
```

---

## Log of Milestones & Experiments

### Milestone 1: Baseline Analysis & Scorer Verification
- **Evaluated:** `3_baseline.py --mode common` on `dev.jsonl`.
- **Finding:** Achieved 5.2% net saving at threshold 1.0 purely by predicting constant boilerplate on repetitive tax forms. Fails on dynamic fields (unfiltered precision 48.9%).

### Milestone 2: Anti-Cheat Governance (`AGENTS.md`)
- Enforced strict rules: No reading `qas` from `CUADv1.json`, no memorizing dev set batch constants.
- Configured SIXT LiteLLM endpoints and model connectivity.

### Milestone 3: Environment & Parsing Infrastructure
- Installed `litellm` and `docling` (with `rapidocr`, `docling-parse`, and PyTorch dependencies).
- Verified live chat completions on SIXT LiteLLM (`gpt-4o-mini` and other models).
- Tested Docling on ExtractBench digital and scanned PDFs: successfully parsed checkboxes (`- [ ]`, `- [x]`) and extracted 8k+ chars of tabular layout from scanned forms that returned 0 chars under PyPDF.

### Milestone 4: Implementation of Two-Stage Node Tree Architecture (`stage1_pipeline/`)
- Active development in `stage1_pipeline/`:
  - `parser.py`: Document classification + Docling/layout parser with hierarchical Node Tree output.
  - `extractor.py`: Schema extraction with `--provider [openai|litellm]` emitting location-only `evidence_span`.

