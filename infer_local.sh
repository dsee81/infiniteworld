#!/bin/bash
# Infinite World - Local Inference Script (Single/Multi GPU)
# Usage: bash infer_local.sh [num_gpus]
# Example: bash infer_local.sh 1   (single GPU, no torchrun, avoids port conflict)
# Example: bash infer_local.sh 8   (8 GPUs via torchrun)
#
# Single GPU (num_gpus=1): runs "python scripts/..." directly, no port needed.
# Multi GPU: runs torchrun. If EADDRINUSE, set: export MASTER_PORT=29500

set -euo pipefail

NUM_GPUS=${1:-1}
# Default: repo root (directory containing this script).
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

cd "$WORK_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# If user didn't activate the env, fall back to `conda run -n infworld ...`.
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "infworld"; then
        PYTHON_BIN="conda run -n infworld python"
        TORCHRUN_BIN="conda run -n infworld torchrun"
    fi
fi

echo "=============================================="
echo "Infinite World - Local Inference"
echo "=============================================="
echo "Using $NUM_GPUS GPU(s)"
echo "Working directory: $WORK_DIR"

$PYTHON_BIN - <<'PY'
import torch
print("torch", getattr(torch, "__version__", "unknown"))
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU: run directly to avoid torchrun port (EADDRINUSE)
    $PYTHON_BIN scripts/infworld_inference.py
else
    MASTER_PORT=${MASTER_PORT:-29400}
    echo "MASTER_PORT: $MASTER_PORT"
    $TORCHRUN_BIN --nnodes=1 --nproc_per_node=$NUM_GPUS \
        --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:$MASTER_PORT \
        scripts/infworld_inference.py
fi
