#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
LLAMA_CACHE_DIR="${LLAMA_CACHE_DIR:-$HOME/.cache/llama.cpp}"

CUE_REPAIR_MODEL_FILE="${CUE_REPAIR_MODEL_FILE:-}"
CUE_REPAIR_MODEL_HF="${CUE_REPAIR_MODEL_HF:-bartowski/google_gemma-3-27b-it-qat-GGUF:Q4_K_M}"
CUE_REPAIR_ALIAS="${CUE_REPAIR_ALIAS:-gemma3-27b}"
CUE_REPAIR_PORT="${CUE_REPAIR_PORT:-8082}"
CUE_REPAIR_CTX="${CUE_REPAIR_CTX:-16384}"
CUE_REPAIR_SEED="${CUE_REPAIR_SEED:-3407}"
CUE_REPAIR_TEMP="${CUE_REPAIR_TEMP:-0}"
CUE_REPAIR_TOP_P="${CUE_REPAIR_TOP_P:-0.9}"
CUE_REPAIR_TOP_K="${CUE_REPAIR_TOP_K:-20}"
CUE_REPAIR_PARALLEL="${CUE_REPAIR_PARALLEL:-1}"
CUE_REPAIR_CACHE_RAM="${CUE_REPAIR_CACHE_RAM:-0}"
CUE_REPAIR_REASONING_FORMAT="${CUE_REPAIR_REASONING_FORMAT:-deepseek}"
CUE_REPAIR_REASONING_BUDGET="${CUE_REPAIR_REASONING_BUDGET:-0}"

if [[ -z "${CUE_REPAIR_MODEL_FILE}" ]]; then
  preferred_pattern="${CUE_REPAIR_MODEL_GLOB:-*gemma-3-27b-it-qat*Q4_K_M*.gguf}"
  while IFS= read -r candidate; do
    CUE_REPAIR_MODEL_FILE="${candidate}"
    break
  done < <(
    find "${LLAMA_CACHE_DIR}" -type f -iname "${preferred_pattern}" 2>/dev/null | sort
    find "${LLAMA_CACHE_DIR}" -type f -iname '*gemma*27b*Q4_K_M*.gguf' 2>/dev/null | sort
  )
fi

if ! command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1; then
  echo "llama-server binary not found: ${LLAMA_SERVER_BIN}" >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -ltnH "( sport = :${CUE_REPAIR_PORT} )" 2>/dev/null | grep -q .; then
  echo "Port ${CUE_REPAIR_PORT} is already in use." >&2
  echo "Set CUE_REPAIR_PORT to a free port before starting the cue repair server." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "Starting Gemma cue repair llama-server"
echo "  bin:       ${LLAMA_SERVER_BIN}"
if [[ -n "${CUE_REPAIR_MODEL_FILE}" ]]; then
  echo "  model:     ${CUE_REPAIR_MODEL_FILE}"
else
  echo "  model:     ${CUE_REPAIR_MODEL_HF}"
fi
echo "  alias:     ${CUE_REPAIR_ALIAS}"
echo "  port:      ${CUE_REPAIR_PORT}"
echo "  ctx:       ${CUE_REPAIR_CTX}"
echo "  seed:      ${CUE_REPAIR_SEED}"
echo "  temp:      ${CUE_REPAIR_TEMP}"
echo "  top-p:     ${CUE_REPAIR_TOP_P}"
echo "  top-k:     ${CUE_REPAIR_TOP_K}"
echo "  np:        ${CUE_REPAIR_PARALLEL}"
echo "  cache:     ${CUE_REPAIR_CACHE_RAM}"
echo "  think fmt: ${CUE_REPAIR_REASONING_FORMAT}"
echo "  think bud: ${CUE_REPAIR_REASONING_BUDGET}"

cmd=(
  "${LLAMA_SERVER_BIN}"
  --alias "${CUE_REPAIR_ALIAS}"
  --port "${CUE_REPAIR_PORT}"
  --ctx-size "${CUE_REPAIR_CTX}"
  --seed "${CUE_REPAIR_SEED}"
  --temp "${CUE_REPAIR_TEMP}"
  --top-p "${CUE_REPAIR_TOP_P}"
  --top-k "${CUE_REPAIR_TOP_K}"
  -np "${CUE_REPAIR_PARALLEL}"
  --cache-ram "${CUE_REPAIR_CACHE_RAM}"
  --reasoning-format "${CUE_REPAIR_REASONING_FORMAT}"
  --reasoning-budget "${CUE_REPAIR_REASONING_BUDGET}"
)

if [[ -n "${CUE_REPAIR_MODEL_FILE}" ]]; then
  cmd+=(--model "${CUE_REPAIR_MODEL_FILE}")
else
  cmd+=(-hf "${CUE_REPAIR_MODEL_HF}")
fi

exec "${cmd[@]}" "$@"
