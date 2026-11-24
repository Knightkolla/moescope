#!/usr/bin/env bash
# Registers / spills / stack per kernel for git revisions, via nvcc -Xptxas -v.
#   bash scripts/ptxas_report.sh <rev>...
set -euo pipefail
cd "$(dirname "$0")/.."
for rev in "$@"; do
  git show "$rev:src/gpu/kernels.rs" > /tmp/kernels_$$.rs
  python3 - /tmp/kernels_$$.rs /tmp/k_$$.cu <<'PY'
import sys
s = open(sys.argv[1]).read()
src = s[s.index('r#"') + 3 : s.index('"#;')]
for a, b in (("MOESCOPE_MAX_K", "32"), ("MOESCOPE_MAX_GROUPS", "32"), ("MOESCOPE_DT", "0"), ("MOESCOPE_SIG", "0")):
    src = src.replace(a, b)
open(sys.argv[2], "w").write(src)
PY
  echo "== $rev"
  nvcc -arch=sm_75 -cubin -o /dev/null -Xptxas -v /tmp/k_$$.cu 2>&1 | grep -E "Compiling entry|registers|spill|stack" | sed 's/ptxas info    : //'
done
