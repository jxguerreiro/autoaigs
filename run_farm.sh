#!/usr/bin/env bash
set -euo pipefail

PY=python3
SCRIPT="main.py"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

gpu_available=false
cudnn9_available=false

# GPU present?
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
  gpu_available=true
fi

# cuDNN 9 present? (required by faster-whisper GPU)
if have ldconfig && ldconfig -p 2>/dev/null | grep -qE 'libcudnn_ops\.so\.9(\.|$)'; then
  cudnn9_available=true
else
  for p in /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9* /usr/local/cuda/lib64/libcudnn_ops.so.9* "${CONDA_PREFIX:-}"/lib/libcudnn_ops.so.9* ; do
    [[ -e "$p" ]] && cudnn9_available=true && break
  done
fi

# Decide device for faster-whisper
if $gpu_available && $cudnn9_available; then
  DEVICE="cuda"
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  fi
else
  DEVICE="cpu"
fi

log "GPU available: $gpu_available | cuDNN9 available: $cudnn9_available | selected device: $DEVICE"

export USE_FASTER_WHISPER="1"
export WHISPER_DEVICE="$DEVICE"
export FASTER_WHISPER_DEVICE="$DEVICE"
export FASTER_WHISPER_COMPUTE_TYPE="$([ "$DEVICE" = "cuda" ] && echo int8_float16 || echo int8)"

# Optional tuning
export MIN_SILENCE_SEC="${MIN_SILENCE_SEC:-0.7}"   # detection threshold
export SILENCE_NOISE_DB="${SILENCE_NOISE_DB:- -35}"

log "Running: $PY $SCRIPT"
time "$PY" "$SCRIPT"
