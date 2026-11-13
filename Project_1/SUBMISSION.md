
# Submission: 

## Authors

**Team name:** Sixth Normal Form (6NF)<your team name></your>

**Project:** Project 1: Teach a machine to say "I don't know"

## Problem and approach

#### Problem

The problem was to improve (or stabilize) the precision of automated information extraction from (unstructured) documents. It includes a tradeoff between human-effort and error caused by automated extractions providing wrong values from documents. If the precision of the automated (via LLMs) extraction is high enough (>99%) it can be seen as an alternative to manual extraction done by humans. Otherwise it can be a more expensive way to do extraction than before, which defeats the purpose of using it in the first place.

#### Approach

Our team (me and some gemini models) took this problem in 2 parts.

1) Parsing Documents
2) Extraction

Before we even start with parsing, an identification process takes place. This is to classify the paper according to its length (in pages mostly). The idea here is that, according to a recent study ([arxiv.org/abs/2607.29677](https://arxiv.org/abs/2607.29677)) VLMs perform the best when the page count is not too high (e.g <= 5 pages). If this is the case, we just pass the pdf to a VLM and start the extraction.

If the document is longer, we need to use LLMs. For parsing documents, best practice approach in the industry seems to be a combination of layout aware OCR and VLMs. To simplify this part, the [Docling](https://docling.ai/) tool was used. It turns both images and PDFs into markdown.

Before this markdown is passed to an LLM, we augment it with location annotations. This is a way to simplify the information grounding of extracted values. Instead of providing a made-up format (or a text sequence) from the original text sequence, LLM can alongside any extracted field, provide the location annotation that it found this data from. This location annotation can then be used to programmatically check that part of the original text ony, lowering costs, speeding up the process and improving judge LLM performance in case an LLM is used for ambigous data that can't just be checked with a text search. Some example annotations are: [sec-1/p-2/s-3], [sec-5/table-0/row-1]...

It is known that confidence levels provided by an LLM tend to be overconfident, especially for positive results. Therefore in our approach we don't trust the confidence level of one LLM but also use a second one (a different model) and run the extraction on it as well.

After this part we end up with two values and 



## Models and why

Any model is allowed, open-weight or closed (API). If your project uses models, show the ones you tried on the same held-out data, and why you chose yours. Add your project-specific settings here too (Project 1: your confidence threshold).

| Model         | Open-weight or closed | Where it runs          | Main metric(s)           | Cost and time per item | Chosen?         |
| ------------- | --------------------- | ---------------------- | ------------------------ | ---------------------- | --------------- |
| <name></name> | <open / closed>       | <local / API provider> | <your project's metrics> | <cost, seconds>        | <yes / no, why> |

## Results

Your numbers, measured on held-out data, with a baseline to compare against (for example a naive model). Use the metrics named in your project README. Say honestly what did not work and what the limits are.

## How to run

Exact commands, from a fresh clone: Python version, install, data location, how to reproduce the results. No secrets: list the environment variables you need (names only).

## Demo

Path to `submission/demo.mp4` (1 min, under 50 MB) **or** `submission/slides.pdf` (max 5 slides).

## Next steps

What you would do with more time, and how SIXT could use this.
