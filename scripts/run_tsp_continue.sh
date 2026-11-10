#!/usr/bin/env bash
# Continue from the latest Qwen TSP submission; pass check/resume/report as needed.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ $# -eq 0 ]]; then
    set -- run
fi
exec uv run --frozen python tsp_continue.py "$@"
