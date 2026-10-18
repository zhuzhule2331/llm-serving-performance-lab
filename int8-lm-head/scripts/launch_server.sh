#!/usr/bin/env bash
# Launch the vLLM server for one stage of the ABBA/BAAB benchmark.
#
# PROVENANCE: this shell script was NOT archived with the original experiment.
# It is reconstructed from the `CMD` list and the environment block inside
# scripts/run_graph_abba.py, which is the script that actually ran the benchmark.
# The reconstructed invocation is identical to that CMD; no flag was added or removed.
#
# Usage:
#   INT8_LM_HEAD_ENABLE=0 bash scripts/launch_server.sh   # BF16 baseline
#   INT8_LM_HEAD_ENABLE=1 bash scripts/launch_server.sh   # INT8 custom path
#
# Run from the project root (int8-lm-head/).

set -euo pipefail

: "${QWEN_MODEL_DIR:?set QWEN_MODEL_DIR to the Qwen3-VL-8B-Instruct snapshot directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-qwen_int8_head}"
export INT8_LM_HEAD_ENABLE="${INT8_LM_HEAD_ENABLE:-0}"
export INT8_LM_HEAD_TRACE="${INT8_LM_HEAD_TRACE:-0}"
export INT8_LM_HEAD_PROFILE="${INT8_LM_HEAD_PROFILE:-0}"
export INT8_LM_HEAD_NSYS="${INT8_LM_HEAD_NSYS:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "INT8_LM_HEAD_ENABLE=${INT8_LM_HEAD_ENABLE}"

exec vllm serve "$QWEN_MODEL_DIR" \
  --served-model-name qwen-int8 \
  --host 127.0.0.1 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85
