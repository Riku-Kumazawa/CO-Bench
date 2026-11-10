#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_python="${COBENCH_VLLM_PYTHON:-/home/rocklee/qwen-vllm/.venv/bin/python}"
model_dir="${COBENCH_MODEL_DIR:-${repo_dir}/models/Qwen3.8-27B}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
tp_size="${COBENCH_TP_SIZE:-4}"
max_model_len="${COBENCH_MAX_MODEL_LEN:-65536}"

if [[ ! -x "$vllm_python" ]]; then
    echo "vLLM Python not found: $vllm_python (set COBENCH_VLLM_PYTHON)." >&2
    exit 1
fi
# An absolute Python path does not activate the virtualenv. FlashInfer invokes
# ninja by name while warming up, so its executable must also be on PATH.
export PATH="$(dirname -- "$vllm_python"):$PATH"
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja not found in the inference environment. Install it with:" >&2
    printf 'uv pip install --python %q ninja\n' "$vllm_python" >&2
    exit 1
fi
if [[ ! -f "$model_dir/checkpoint.json" ]]; then
    echo "Checkpoint is not ready. Run: uv run --frozen python scripts/prepare_qwen.py" >&2
    exit 1
fi
IFS=',' read -r -a devices <<< "$gpu_ids"
if [[ ${#devices[@]} -ne "$tp_size" ]]; then
    echo "CUDA_VISIBLE_DEVICES must list exactly COBENCH_TP_SIZE=$tp_size GPUs." >&2
    exit 1
fi

command=(
    "$vllm_python" -m vllm.entrypoints.openai.api_server
    --model "$model_dir"
    --served-model-name Qwen/Qwen3.8-27B
    --host 127.0.0.1 --port "${COBENCH_PORT:-8000}"
    --dtype bfloat16 --tensor-parallel-size "$tp_size"
    --distributed-executor-backend mp
    --language-model-only --reasoning-parser qwen3
    --max-model-len "$max_model_len" --max-num-seqs 1
    --gpu-memory-utilization 0.9
    --generation-config vllm
)
if [[ "${1:-}" == "--dry-run" && $# -eq 1 ]]; then
    printf 'PATH=%q CUDA_VISIBLE_DEVICES=%q ' "$PATH" "$gpu_ids"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_ids"
exec "${command[@]}"
