#!/usr/bin/env bash
# Real-vLLM check for moescope.vllm, in its own venv so vLLM's torch does not
# replace the system one. Needs a built wheel in dist2/ (or dist/).
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q uv
# Kaggle / Colab set UV_SYSTEM_PYTHON=true, which makes `uv pip` ignore the
# venv and overwrite the system torch: always name the venv's interpreter.
export UV_SYSTEM_PYTHON=false
uv venv -q /tmp/vllm-env --python "$(command -v python3)"
PY=/tmp/vllm-env/bin/python
uv pip install -q --python "$PY" vllm
uv pip install -q --python "$PY" --no-deps "$(ls dist2/moescope-*.whl dist/moescope-*.whl 2>/dev/null | head -1)"
source /tmp/vllm-env/bin/activate
python -c "import vllm, torch, moescope; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', moescope.cuda_available())"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python scripts/vllm_check.py "$@"
