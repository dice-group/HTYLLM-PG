#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CKPT="${REPO_ROOT}/outputs/local_cola_lpr_2gpu/cola_lpr_20260106_145913/checkpoint-282"
BASE_MODEL="hf-internal-testing/tiny-random-LlamaForCausalLM"

python3 "${REPO_ROOT}/scripts/merge_adapter_shards.py" \
  --adapter-sharded-dir "${CKPT}_adapter_sharded" \
  --output-dir "${CKPT}_adapter" \
  --base-model "${BASE_MODEL}"
