#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-qwen-align}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Creating Qwen aligner environment"
echo "  python: ${PYTHON_BIN}"
echo "  venv:   ${VENV_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "git+https://github.com/femelo/py-qwen3-asr-cpp.git"

echo
echo "Done."
echo "Use this interpreter with scripts/align_qwen_forced.py:"
echo "  ${VENV_DIR}/bin/python"
