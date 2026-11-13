# Submission:

## Authors

**Team name:** Sixth Normal Form (6NF)<your team name></your>

**Project:** Project 1: Teach a machine to say "I don't know"

## Problem and approach

#### Problem

The problem was to improve (or stabilize) the precision of automated information extraction from (unstructured) documents. It includes a tradeoff between human-effort and error caused by automated extractions providing wrong values from documents. If the precision of the automated (via LLMs) extraction is high enough (>99%) it can be seen as an alternative to manual extraction done by humans. Otherwise it can be a more expensive way to do extraction than before, which defeats the purpose of using it in the first place.
Automated information extraction from unstructured documents (contracts, scanned forms, PDFs, and images) involves an asymmetric tradeoff between human review effort and the cost of silent errors. Under the evaluation cost model, sending an uncertain field to a human costs **1**, whereas accepting a wrong or hallucinated value automatically costs **100**.

Because of this 100:1 penalty, raw extraction accuracy is not enough: automated extraction only beats manual entry when accepted predictions achieve **>99% precision**. If a system is overconfident and lets even a small number of errors slip through, it becomes more expensive than doing the work by hand. The core challenge is therefore **selective prediction**—teaching the system when to say *"I don't know"* and defer to a human.

#### Approach

Our team (me and some gemini models) split the pipeline into multiple stages:

1) **Modality Identification & Adaptive Routing**
   Before parsing, each document is classified by type and length (e.g., page count and whether it is a digital PDF, scan, or image). Short digital PDFs ($\le 5$ pages) and image/scanned pages are routed directly to a Vision-Language Model (VLM), since VLM accuracy degrades on very long multi-page documents ([arxiv.org/abs/2607.29677](https://arxiv.org/abs/2607.29677)). Longer documents are routed to the text pipeline.
2) **Layout-Aware Parsing & Location Annotations**
   For longer documents, we parse PDFs and images into structured markdown and tables using [Docling](https://docling.ai/) (with `pypdf`/`pdftotext` fallbacks when needed). Before passing this text to an LLM, we build a  node tree and annotate every block with deterministic location tags such as `[sec-1/p-2/s-3]` or `[sec-5/table-0/row-1]`.
   Instead of relying on free-form quotes or brittle character offsets, the LLM returns the location tag alongside each extracted value. This lets our verifiers programmatically inspect only the cited region—reducing token cost, speeding up verification, and improving accuracy when a verifier or judge checks ambiguous fields. Large schemas (20+ fields) are also chunked to prevent attention degradation. You can think of this as the word boxes in VLM/OCR grounding.
3) **Dual-Model Extraction, Grounding & Confidence Gating**
   LLMs tend to be overconfident—especially on positive extractions—so we do not trust a single model's self-reported confidence. Instead, we run a second, independent model pass (or adversarial critic) to cross-check candidate extractions:

- **Grounding check:** Verify that the cited location tag exists and that the extracted value is grounded in that section's source text.
- **Cross-model consensus:** Compare predictions from both models. Agreement with verified grounding boosts confidence (`>= 0.97`), while disagreement, failed grounding, or missing evidence sharply penalizes it.
- **Confidence thresholding ("I don't know"):** Only values above the calibrated threshold (`0.95`) are accepted automatically. Anything below the threshold—or any field absent from the document—is emitted as `null` / low confidence and routed to human review.

## Models and why

| Model                 | Open-weight or closed | Where it runs                                 | Chosen?                                                                                                                 |
| --------------------- | --------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Gemini 3.5 Flash Lite | closed                | Sixt LiteLLM (Vertex AI) and Google AI Studio | Fast, Not bad multimodal performance                                                                                    |
| GPT 5.5               | closed                | Sixt LiteLLM                                  | Was one of the best in https://arxiv.org/abs/2607.29677                                                                 |
| GPT 4o-mini           | closed                | Sixt LiteLLM                                  | Cheaper than GPT 5.5, provides a good candidate for dual consensus setup due to different behavior with the other model |

## Results

Your numbers, measured on held-out data, with a baseline to compare against (for example a naive model). Use the metrics named in your project README. Say honestly what did not work and what the limits are.

## How to run

### Prerequisites

- **Python:** 3.11+ (project environment uses `>=3.14` in `pyproject.toml`)
- **Dataset & setup:** Run `setup.sh` from a fresh clone to create `.venv`, install `requirements.txt`, download raw data, and prepare `data/unified/` JSONL files:

```bash
bash setup.sh --accept-licenses
. .venv/bin/activate
```

### Environment variables

Copy `.env.example` to `.env` and set your credentials (never commit `.env`):

- SIXT LiteLLM endpoint: `SIXT_LLM_KEY`, `SIXT_LLM_URL` (and optionally `SIXT_LLM_MODEL`)
- Or OpenAI-compatible / Gemini endpoint: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Optional logging: `LOG_LEVEL`, `PIPELINE_LOG_FILE`

### Run extraction and score

1. **Run the pipeline on a dataset (`stage1_pipeline/extractor.py`):**

```bash
.venv/bin/python3 stage1_pipeline/extractor.py \
    --data data/unified/dev_mini.jsonl \
    --out preds.jsonl \
    --concurrency 10
```

*(For an offline regex/keyword fallback with zero LLM calls, add `--mode offline`.)*

2. **Compare strategies side-by-side (`benchmark.py`):**

```bash
.venv/bin/python3 benchmark.py \
    --data data/unified/dev_mini.jsonl \
    --configs configs/m1_routed_single.json configs/m2_dual_consensus.json configs/m3_critic_verified.json \
    --out-dir runs/
```

3. **Score predictions (`5_score.py`):**

```bash
.venv/bin/python3 5_score.py \
    --gold data/unified/dev_mini.jsonl \
    --pred preds.jsonl \
    --threshold 0.95 \
    --error-cost 100
```

4. **Run the hidden script (`run_hidden.sh`):**

```bash
bash run_hidden.sh <input.jsonl> <output.jsonl>
```

For example, to test on a small local dataset:

```bash
bash run_hidden.sh data/unified/dev_mini2.jsonl runs/hidden_preds.jsonl
```

This runs `benchmark.py` with `configs/m2_dual_consensus.json`, writes predictions to `<output.jsonl>`, and passes `--force` to overwrite any existing checkpoint predictions. Both `<input.jsonl>` and `<output.jsonl>` arguments are required (`set -euo pipefail` will error if either is omitted).

We support providing multiple options via config files. More information can be found at [USAGE.md](USAGE.md)

## Demo

[slides.pdf](submission/slides.pdf)

## Next steps

One of the problems with having a time constraint like this is to test different configurations. There are many options such as `penalty_mul`, `risk_json_mult`, `vlm_max_pages` and so on that highly effects the performance. Model selection is another topic. Results indicate that this solution can be a viable option for this task but requires careful considerations across different domains (costs, latency, coverage etc.)
