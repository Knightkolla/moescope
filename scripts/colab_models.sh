#!/usr/bin/env bash
# Builds moescope on a Colab T4 (or any Linux + NVIDIA box), then runs the GPU
# stress test and the routing report on real MoE models.
#
#   bash scripts/colab_models.sh                 # cuda-13000, default model list
#   CUDA_FEATURE=cuda-12080 bash scripts/colab_models.sh model1 model2
#
# Results land in results/<model>/ and bench/results/stress_t4.json.
set -euo pipefail
FEATURE="${CUDA_FEATURE:-cuda-13000}"
cd "$(dirname "$0")/.."
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(ibm-granite/granite-3.1-3b-a800m-base allenai/OLMoE-1B-7B-0924)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

pip install -q maturin matplotlib accelerate bitsandbytes pytest
rm -rf dist
maturin build --release -F "python,pyo3/extension-module,$FEATURE" -o dist
pip install -q --force-reinstall --no-deps dist/moescope-*.whl
python -c "import moescope; assert moescope.cuda_available(); print('moescope CUDA backend OK')"

echo "== attach() parity on the GPU kernel (tiny real-architecture models)"
MOESCOPE_DEVICE=cuda python -m pytest -q tests/python/test_hf_attach.py tests/python/test_cuda_streams.py

echo "== stress test"
python bench/stress_test.py --out bench/results/stress_t4.json

for m in "${MODELS[@]}"; do
  echo "== $m"
  flags=""
  # fp16 LFM2-8B (17 GB) is split over Kaggle's two T4s
  case "$m" in *LFM2-8B*) flags="--device-map auto";; esac
  python examples/analyze_routing.py --model "$m" --out "results/${m##*/}" $flags
  rm -rf ~/.cache/huggingface/hub/models--"${m//\//--}"  # free disk for the next model
done
echo "ALL RUNS DONE"
