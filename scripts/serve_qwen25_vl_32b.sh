#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/root/highspeedstorage/test_storage/sbingqua/projects/host_local_llms/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-/root/highspeedstorage/test_storage/sbingqua/projects/host_local_llms/.venv/bin/vllm}"
VLLM_BIN_DIR="$(cd "$(dirname "$VLLM_BIN")" && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
AUTO_SELECT_GPU="${AUTO_SELECT_GPU:-1}"

if [[ ! -x "$VLLM_PYTHON_BIN" ]]; then
  echo "vLLM Python binary not found: $VLLM_PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM CLI binary not found: $VLLM_BIN" >&2
  exit 1
fi

export HF_HOME
export PATH="$VLLM_BIN_DIR:$PATH"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$AUTO_SELECT_GPU" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2,2n \
        | head -n 1 \
        | cut -d, -f1 \
        | tr -d ' '
    )"
    export CUDA_VISIBLE_DEVICES
    echo "Auto-selected GPU: $CUDA_VISIBLE_DEVICES" >&2
  fi
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS"
