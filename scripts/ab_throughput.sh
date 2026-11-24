#!/usr/bin/env bash
# A/B kernel throughput on the same GPU and session: builds two git revisions
# of moescope and times one route_and_log call (float32 softmax, 1M tokens).
#   bash scripts/ab_throughput.sh <rev-a> <rev-b>
set -euo pipefail
FEATURE="${CUDA_FEATURE:-cuda-13000}"
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env"
for rev in "$@"; do
  rm -rf /tmp/ab && git worktree add -f -q /tmp/ab "$rev"
  (cd /tmp/ab && maturin build -q --release -F "python,pyo3/extension-module,$FEATURE" -o /tmp/ab/dist >/dev/null 2>&1)
  pip install -q --force-reinstall --no-deps /tmp/ab/dist/moescope-*.whl
  python - "$rev" <<'PY'
import sys, time, torch, moescope
rev = sys.argv[1]
for e, k in ((8, 2), (32, 8), (64, 8), (128, 8)):
    n = 1 << 20
    x = torch.randn(n, e, device="cuda")
    moescope.configure(capacity=60 * n * k, mode="softmax_topk", device="cuda")
    for _ in range(5):
        moescope.route_and_log(x, k=k, layer_id=0)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50):
        moescope.route_and_log(x, k=k, layer_id=0)
    torch.cuda.synchronize()
    print(f"{rev[:8]} E={e:<4} k={k}: {(time.perf_counter() - t) / 50 * 1e3:7.3f} ms", flush=True)
    moescope.configure(capacity=1, device="cuda"); del x; torch.cuda.empty_cache()
PY
  git worktree remove -f /tmp/ab
done
