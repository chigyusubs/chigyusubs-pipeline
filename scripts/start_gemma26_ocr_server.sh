#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
GEMMA26_OCR_MODEL="${GEMMA26_OCR_MODEL:-bartowski/google_gemma-4-26B-A4B-it-GGUF:IQ4_XS}"
GEMMA26_OCR_ALIAS="${GEMMA26_OCR_ALIAS:-gemma-4-26B-A4B-it-IQ4_XS}"
GEMMA26_OCR_PORT="${GEMMA26_OCR_PORT:-8000}"
GEMMA26_OCR_CTX="${GEMMA26_OCR_CTX:-8192}"
GEMMA26_OCR_SEED="${GEMMA26_OCR_SEED:-3407}"
GEMMA26_OCR_TEMP="${GEMMA26_OCR_TEMP:-1}"
GEMMA26_OCR_TOP_P="${GEMMA26_OCR_TOP_P:-0.95}"
GEMMA26_OCR_TOP_K="${GEMMA26_OCR_TOP_K:-64}"
GEMMA26_OCR_PARALLEL="${GEMMA26_OCR_PARALLEL:-1}"
GEMMA26_OCR_CACHE_RAM="${GEMMA26_OCR_CACHE_RAM:-0}"

if ! command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1; then
  echo "llama-server binary not found: ${LLAMA_SERVER_BIN}" >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -ltnH "( sport = :${GEMMA26_OCR_PORT} )" 2>/dev/null | grep -q .; then
  echo "Port ${GEMMA26_OCR_PORT} is already in use." >&2
  echo "Set GEMMA26_OCR_PORT to a free port before starting the Gemma 26B OCR server." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "Starting Gemma 26B OCR llama-server"
echo "  bin:    ${LLAMA_SERVER_BIN}"
echo "  model:  ${GEMMA26_OCR_MODEL}"
echo "  alias:  ${GEMMA26_OCR_ALIAS}"
echo "  port:   ${GEMMA26_OCR_PORT}"
echo "  ctx:    ${GEMMA26_OCR_CTX}"
echo "  seed:   ${GEMMA26_OCR_SEED}"
echo "  temp:   ${GEMMA26_OCR_TEMP}"
echo "  top-p:  ${GEMMA26_OCR_TOP_P}"
echo "  top-k:  ${GEMMA26_OCR_TOP_K}"
echo "  np:     ${GEMMA26_OCR_PARALLEL}"
echo "  cache:  ${GEMMA26_OCR_CACHE_RAM}"

exec "${LLAMA_SERVER_BIN}" \
  -hf "${GEMMA26_OCR_MODEL}" \
  --alias "${GEMMA26_OCR_ALIAS}" \
  --port "${GEMMA26_OCR_PORT}" \
  --ctx-size "${GEMMA26_OCR_CTX}" \
  --seed "${GEMMA26_OCR_SEED}" \
  --temp "${GEMMA26_OCR_TEMP}" \
  --top-p "${GEMMA26_OCR_TOP_P}" \
  --top-k "${GEMMA26_OCR_TOP_K}" \
  -np "${GEMMA26_OCR_PARALLEL}" \
  --cache-ram "${GEMMA26_OCR_CACHE_RAM}" \
  "$@"
