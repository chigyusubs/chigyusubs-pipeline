#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
LLAMA_CACHE_DIR="${LLAMA_CACHE_DIR:-$HOME/.cache/llama.cpp}"
QWEN_OCR_MODEL="${QWEN_OCR_MODEL:-unsloth/Qwen3.5-9B-GGUF:Q6_K}"
QWEN_OCR_MMPROJ="${QWEN_OCR_MMPROJ:-}"
QWEN_OCR_ALIAS="${QWEN_OCR_ALIAS:-qwen3.5-9b}"
QWEN_OCR_PORT="${QWEN_OCR_PORT:-8787}"
QWEN_OCR_CTX="${QWEN_OCR_CTX:-8192}"
QWEN_OCR_SEED="${QWEN_OCR_SEED:-3407}"
QWEN_OCR_TEMP="${QWEN_OCR_TEMP:-0}"
QWEN_OCR_TOP_P="${QWEN_OCR_TOP_P:-0.9}"
QWEN_OCR_TOP_K="${QWEN_OCR_TOP_K:-20}"
QWEN_OCR_PARALLEL="${QWEN_OCR_PARALLEL:-1}"
QWEN_OCR_CACHE_RAM="${QWEN_OCR_CACHE_RAM:-0}"

if [[ -z "${QWEN_OCR_MMPROJ}" ]]; then
  preferred_pattern="${QWEN_OCR_MMPROJ_GLOB:-*Qwen3.5-9B*mmproj*.gguf}"
  while IFS= read -r candidate; do
    QWEN_OCR_MMPROJ="${candidate}"
    break
  done < <(
    find "${LLAMA_CACHE_DIR}" -type f -name "${preferred_pattern}" 2>/dev/null | sort
    find "${LLAMA_CACHE_DIR}" -type f -name '*mmproj*.gguf' 2>/dev/null | sort
  )
fi

if [[ -z "${QWEN_OCR_MMPROJ}" ]]; then
  echo "Could not auto-detect mmproj under ${LLAMA_CACHE_DIR}." >&2
  echo "Set QWEN_OCR_MMPROJ explicitly before starting the OCR server." >&2
  exit 1
fi

if ! command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1; then
  echo "llama-server binary not found: ${LLAMA_SERVER_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "Starting OCR llama-server"
echo "  bin:    ${LLAMA_SERVER_BIN}"
echo "  model:  ${QWEN_OCR_MODEL}"
echo "  mmproj: ${QWEN_OCR_MMPROJ}"
echo "  alias:  ${QWEN_OCR_ALIAS}"
echo "  port:   ${QWEN_OCR_PORT}"
echo "  ctx:    ${QWEN_OCR_CTX}"
echo "  seed:   ${QWEN_OCR_SEED}"
echo "  temp:   ${QWEN_OCR_TEMP}"
echo "  top-p:  ${QWEN_OCR_TOP_P}"
echo "  top-k:  ${QWEN_OCR_TOP_K}"
echo "  np:     ${QWEN_OCR_PARALLEL}"
echo "  cache:  ${QWEN_OCR_CACHE_RAM}"

exec "${LLAMA_SERVER_BIN}" \
  -hf "${QWEN_OCR_MODEL}" \
  --mmproj "${QWEN_OCR_MMPROJ}" \
  --alias "${QWEN_OCR_ALIAS}" \
  --port "${QWEN_OCR_PORT}" \
  --ctx-size "${QWEN_OCR_CTX}" \
  --seed "${QWEN_OCR_SEED}" \
  --temp "${QWEN_OCR_TEMP}" \
  --top-p "${QWEN_OCR_TOP_P}" \
  --top-k "${QWEN_OCR_TOP_K}" \
  -np "${QWEN_OCR_PARALLEL}" \
  --cache-ram "${QWEN_OCR_CACHE_RAM}" \
  "$@"
