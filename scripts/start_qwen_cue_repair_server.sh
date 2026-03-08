#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
LLAMA_CACHE_DIR="${LLAMA_CACHE_DIR:-$HOME/.cache/llama.cpp}"

QWEN_CUE_REPAIR_MODEL_FILE="${QWEN_CUE_REPAIR_MODEL_FILE:-}"
QWEN_CUE_REPAIR_MODEL_HF="${QWEN_CUE_REPAIR_MODEL_HF:-bartowski/Qwen_Qwen3.5-35B-A3B-GGUF:Q4_K_M}"
QWEN_CUE_REPAIR_ALIAS="${QWEN_CUE_REPAIR_ALIAS:-qwen3.5-35b-a3b}"
QWEN_CUE_REPAIR_PORT="${QWEN_CUE_REPAIR_PORT:-8083}"
QWEN_CUE_REPAIR_CTX="${QWEN_CUE_REPAIR_CTX:-16384}"
QWEN_CUE_REPAIR_SEED="${QWEN_CUE_REPAIR_SEED:-3407}"
QWEN_CUE_REPAIR_TEMP="${QWEN_CUE_REPAIR_TEMP:-0}"
QWEN_CUE_REPAIR_TOP_P="${QWEN_CUE_REPAIR_TOP_P:-0.8}"
QWEN_CUE_REPAIR_TOP_K="${QWEN_CUE_REPAIR_TOP_K:-20}"
QWEN_CUE_REPAIR_PARALLEL="${QWEN_CUE_REPAIR_PARALLEL:-1}"
QWEN_CUE_REPAIR_CACHE_RAM="${QWEN_CUE_REPAIR_CACHE_RAM:-0}"
QWEN_CUE_REPAIR_REASONING_FORMAT="${QWEN_CUE_REPAIR_REASONING_FORMAT:-deepseek}"
QWEN_CUE_REPAIR_REASONING_BUDGET="${QWEN_CUE_REPAIR_REASONING_BUDGET:-0}"

if [[ -z "${QWEN_CUE_REPAIR_MODEL_FILE}" ]]; then
  preferred_pattern="${QWEN_CUE_REPAIR_MODEL_GLOB:-*Qwen*3.5*35B*A3B*Q4_K_M*.gguf}"
  while IFS= read -r candidate; do
    QWEN_CUE_REPAIR_MODEL_FILE="${candidate}"
    break
  done < <(
    find "${LLAMA_CACHE_DIR}" -type f -iname "${preferred_pattern}" 2>/dev/null | sort
    find "${LLAMA_CACHE_DIR}" -type f -iname '*qwen*35b*a3b*Q4_K_M*.gguf' 2>/dev/null | sort
  )
fi

if ! command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1; then
  echo "llama-server binary not found: ${LLAMA_SERVER_BIN}" >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -ltnH "( sport = :${QWEN_CUE_REPAIR_PORT} )" 2>/dev/null | grep -q .; then
  echo "Port ${QWEN_CUE_REPAIR_PORT} is already in use." >&2
  echo "Set QWEN_CUE_REPAIR_PORT to a free port before starting the Qwen cue repair server." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "Starting Qwen cue repair llama-server"
echo "  bin:       ${LLAMA_SERVER_BIN}"
if [[ -n "${QWEN_CUE_REPAIR_MODEL_FILE}" ]]; then
  echo "  model:     ${QWEN_CUE_REPAIR_MODEL_FILE}"
else
  echo "  model:     ${QWEN_CUE_REPAIR_MODEL_HF}"
fi
echo "  alias:     ${QWEN_CUE_REPAIR_ALIAS}"
echo "  port:      ${QWEN_CUE_REPAIR_PORT}"
echo "  ctx:       ${QWEN_CUE_REPAIR_CTX}"
echo "  seed:      ${QWEN_CUE_REPAIR_SEED}"
echo "  temp:      ${QWEN_CUE_REPAIR_TEMP}"
echo "  top-p:     ${QWEN_CUE_REPAIR_TOP_P}"
echo "  top-k:     ${QWEN_CUE_REPAIR_TOP_K}"
echo "  np:        ${QWEN_CUE_REPAIR_PARALLEL}"
echo "  cache:     ${QWEN_CUE_REPAIR_CACHE_RAM}"
echo "  think fmt: ${QWEN_CUE_REPAIR_REASONING_FORMAT}"
echo "  think bud: ${QWEN_CUE_REPAIR_REASONING_BUDGET}"

cmd=(
  "${LLAMA_SERVER_BIN}"
  --alias "${QWEN_CUE_REPAIR_ALIAS}"
  --port "${QWEN_CUE_REPAIR_PORT}"
  --ctx-size "${QWEN_CUE_REPAIR_CTX}"
  --seed "${QWEN_CUE_REPAIR_SEED}"
  --temp "${QWEN_CUE_REPAIR_TEMP}"
  --top-p "${QWEN_CUE_REPAIR_TOP_P}"
  --top-k "${QWEN_CUE_REPAIR_TOP_K}"
  -np "${QWEN_CUE_REPAIR_PARALLEL}"
  --cache-ram "${QWEN_CUE_REPAIR_CACHE_RAM}"
  --reasoning-format "${QWEN_CUE_REPAIR_REASONING_FORMAT}"
  --reasoning-budget "${QWEN_CUE_REPAIR_REASONING_BUDGET}"
)

if [[ -n "${QWEN_CUE_REPAIR_MODEL_FILE}" ]]; then
  cmd+=(--model "${QWEN_CUE_REPAIR_MODEL_FILE}")
else
  cmd+=(-hf "${QWEN_CUE_REPAIR_MODEL_HF}")
fi

exec "${cmd[@]}" "$@"
