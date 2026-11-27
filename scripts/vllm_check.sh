#!/usr/bin/env bash
# Real-vLLM check for moescope.vllm, in its own venv so vLLM's torch does not
# replace the system one. Needs a built wheel in dist2/ (or dist/).
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q uv
uv venv -q /tmp/vllm-env --python "$(command -v python3)"
source /tmp/vllm-env/bin/activate
uv pip install -q vllm
uv pip install -q --no-deps "$(ls dist2/moescope-*.whl dist/moescope-*.whl 2>/dev/null | head -1)"
python -c "import vllm, torch, moescope; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', moescope.cuda_available())"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python scripts/vllm_check.py "$@"
