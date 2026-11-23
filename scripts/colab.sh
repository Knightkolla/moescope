#!/usr/bin/env bash
# Runs every phase's checks on a Colab GPU runtime (T4) or any Linux+NVIDIA box.
#
#   bash scripts/colab.sh               # uses cuda-13000
#   bash scripts/colab.sh cuda-12080    # match `nvidia-smi` "CUDA Version"
#
# Success = the script reaches "ALL PHASES PASSED" (it stops at the first failure).
set -euo pipefail
FEATURE="${1:-${CUDA_FEATURE:-cuda-13000}}"
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"
# libcuda is on the default path on Colab; NVRTC ships with the toolkit.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# Make GPU tests fail (not skip) if the driver can't be loaded.
export MOESCOPE_REQUIRE_GPU=1

echo "== Phase 2: vector_add smoke test"
cargo run --release --example vector_add --features "$FEATURE"

echo "== Phases 1-4: Rust tests, CPU + GPU"
cargo test --release --features "$FEATURE"

echo "== Phase 5: Python bindings"
pip install -q maturin pytest
rm -rf dist && # -F replaces pyproject features, so list python + extension-module again.
maturin build --release -F "python,pyo3/extension-module,$FEATURE" -o dist
pip install -q --force-reinstall --no-deps dist/moescope-*.whl
python -c "import moescope; assert moescope.cuda_available(), 'built without CUDA or no driver'; print('moescope CUDA backend OK')"
MOESCOPE_DEVICE=cuda python -m pytest -q tests/python
python examples/hf_moe_hook.py --arch olmoe --device cuda
python examples/hf_moe_hook.py --arch mixtral --device cuda --layer 1

echo "== Phase 6: benchmark"
python bench/bench_hook.py --device cuda --out bench/results/t4.json

echo "ALL PHASES PASSED"
