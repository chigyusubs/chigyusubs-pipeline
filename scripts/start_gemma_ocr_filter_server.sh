#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
LLAMA_CACHE_DIR="${LLAMA_CACHE_DIR:-$HOME/.cache/llama.cpp}"
GEMMA_FILTER_MODEL_FILE="${GEMMA_FILTER_MODEL_FILE:-}"
GEMMA_FILTER_MODEL_HF="${GEMMA_FILTER_MODEL_HF:-bartowski/google_gemma-3-27b-it-qat-GGUF:Q4_K_M}"
GEMMA_FILTER_ALIAS="${GEMMA_FILTER_ALIAS:-gemma3-27b}"
GEMMA_FILTER_PORT="${GEMMA_FILTER_PORT:-8081}"
GEMMA_FILTER_CTX="${GEMMA_FILTER_CTX:-8192}"
GEMMA_FILTER_SEED="${GEMMA_FILTER_SEED:-3407}"
GEMMA_FILTER_TEMP="${GEMMA_FILTER_TEMP:-0}"
GEMMA_FILTER_TOP_P="${GEMMA_FILTER_TOP_P:-0.9}"
GEMMA_FILTER_TOP_K="${GEMMA_FILTER_TOP_K:-20}"
GEMMA_FILTER_PARALLEL="${GEMMA_FILTER_PARALLEL:-1}"
GEMMA_FILTER_CACHE_RAM="${GEMMA_FILTER_CACHE_RAM:-0}"

if [[ -z "${GEMMA_FILTER_MODEL_FILE}" ]]; then
  preferred_pattern="${GEMMA_FILTER_MODEL_GLOB:-*gemma-3-27b-it-qat*Q4_K_M*.gguf}"
  while IFS= read -r candidate; do
    GEMMA_FILTER_MODEL_FILE="${candidate}"
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

if command -v ss >/dev/null 2>&1 && ss -ltnH "( sport = :${GEMMA_FILTER_PORT} )" 2>/dev/null | grep -q .; then
  echo "Port ${GEMMA_FILTER_PORT} is already in use." >&2
  echo "Set GEMMA_FILTER_PORT to a free port before starting the Gemma filter server." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "Starting Gemma OCR filter llama-server"
echo "  bin:    ${LLAMA_SERVER_BIN}"
if [[ -n "${GEMMA_FILTER_MODEL_FILE}" ]]; then
  echo "  model:  ${GEMMA_FILTER_MODEL_FILE}"
else
  echo "  model:  ${GEMMA_FILTER_MODEL_HF}"
fi
echo "  alias:  ${GEMMA_FILTER_ALIAS}"
echo "  port:   ${GEMMA_FILTER_PORT}"
echo "  ctx:    ${GEMMA_FILTER_CTX}"
echo "  seed:   ${GEMMA_FILTER_SEED}"
echo "  temp:   ${GEMMA_FILTER_TEMP}"
echo "  top-p:  ${GEMMA_FILTER_TOP_P}"
echo "  top-k:  ${GEMMA_FILTER_TOP_K}"
echo "  np:     ${GEMMA_FILTER_PARALLEL}"
echo "  cache:  ${GEMMA_FILTER_CACHE_RAM}"

cmd=(
  "${LLAMA_SERVER_BIN}"
  --alias "${GEMMA_FILTER_ALIAS}"
  --port "${GEMMA_FILTER_PORT}"
  --ctx-size "${GEMMA_FILTER_CTX}"
  --seed "${GEMMA_FILTER_SEED}"
  --temp "${GEMMA_FILTER_TEMP}"
  --top-p "${GEMMA_FILTER_TOP_P}"
  --top-k "${GEMMA_FILTER_TOP_K}"
  -np "${GEMMA_FILTER_PARALLEL}"
  --cache-ram "${GEMMA_FILTER_CACHE_RAM}"
)

if [[ -n "${GEMMA_FILTER_MODEL_FILE}" ]]; then
  cmd+=(--model "${GEMMA_FILTER_MODEL_FILE}")
else
  cmd+=(-hf "${GEMMA_FILTER_MODEL_HF}")
fi

exec "${cmd[@]}" "$@"
