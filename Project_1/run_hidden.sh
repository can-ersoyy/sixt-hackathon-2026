#!/usr/bin/env bash
# Called as: bash run_hidden.sh <input.jsonl> <output.jsonl>
set -euo pipefail

if [ -x ".venv/bin/python3" ]; then
  PYTHON=".venv/bin/python3"
else
  PYTHON="python3"
fi

"$PYTHON" benchmark.py \
  --data "$1" \
  --configs configs/m2_dual_consensus.json \
  --out "$2" \
  --force
