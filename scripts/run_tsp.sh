#!/usr/bin/env bash
# Run one TSP candidate with the same settings as the Aircraft landing run.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

exec uv run --frozen python main.py \
  --task "Travelling salesman problem" \
  --model "Qwen/Qwen3.8-27B" \
  --thinking \
  --reasoning-effort medium \
  --max-tokens 32768 \
  --generation-timeout 7200 \
  --server-wait-timeout 600 \
  --timeout 10 \
  --cpu-num 1 \
  --seed 0 \
  "$@"
