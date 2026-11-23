# moescope

MoE routing telemetry in Rust. moescope routes tokens to experts (top-k plus gate weights) and logs every decision, `(token_id, layer_id, expert_id, gate)`, into a fixed-size buffer in **GPU memory, written by the routing kernel itself**. The whole log is drained to the host in one `drain()` call.

## What this is / is not

**It is:**
- A way to record *every* routing decision of an MoE model (expert load, dead experts, gate distributions per layer) without a host sync per step.
- A CPU reference router and dispatcher. The CUDA kernels are tested against it: exact expert ids, gates within 1e-5.
- A fused CUDA kernel (NVRTC, compiled at runtime) that does route, per-expert dispatch (atomic slots), the load histogram and the telemetry append in one pass.
- Python bindings (`route_and_log`, `drain` returning a numpy structured array), zero-copy from torch CUDA tensors.

**It is not:**
- A faster MoE router. It doesn't compete on throughput. It *recomputes* routing from the logits instead of reusing the model's own top-k, and it allocates per call.
- A replacement for the model's router. It observes; it doesn't change the forward pass.
- Bit-identical to torch near exact ties. Ties go to the lower expert index, and selection runs on logits rather than probabilities.

## Layout

| path | what |
|---|---|
| `src/router.rs` | `Route`, `GateMode` (`SoftmaxThenTopK` = OLMoE/Switch, `TopKThenSoftmax` = Mixtral), `route_batch` → flat `[token*k + j]` |
| `src/dispatcher.rs` | `DispatchPlan`: per-expert token groups + load histogram |
| `src/telemetry.rs` | `#[repr(C)] TelemetryRecord` (12 B), `HostTelemetry` (bounded, counts drops) |
| `src/gpu/` (feature `cuda`) | `GpuRouter`, `TelemetryBuffer::new(&router, cap)` / `drain()`, kernels in `kernels.rs` |
| `src/gpu/raw.rs` | **the only module allowed `unsafe`** (crate is `#![deny(unsafe_code)]`); each block has a SAFETY comment |
| `src/python.rs`, `python/moescope/` | PyO3 bindings (feature `python`, built with maturin) |
| `examples/` | `vector_add.rs` (CUDA smoke test), `hf_moe_hook.py` (HF OLMoE/Mixtral parity check) |
| `bench/bench_hook.py` | logging-overhead benchmark |

## Build and test

Local (macOS, no CUDA):

```sh
cargo test                                  # CPU tests
cargo test --features cuda-13000            # also compiles the GPU path; GPU tests print SKIP
uv venv .venv && source .venv/bin/activate && uv pip install maturin numpy torch transformers pytest
maturin develop --release && python -m pytest -q tests/python
python examples/hf_moe_hook.py --arch olmoe # ends with "OK: all ... match the model's router"
```

GPU (Colab T4 or any Linux + NVIDIA machine): `bash scripts/colab.sh [cuda-13000|cuda-12080|cuda-12040|cuda-auto]`. Pick the feature that matches the "CUDA Version" shown by `nvidia-smi`. The script exits at the first failure and ends with `ALL PHASES PASSED` on success. It sets `MOESCOPE_REQUIRE_GPU=1`, so a missing driver fails the tests instead of skipping them.

## Python

```python
import moescope
moescope.configure(capacity=1 << 24, mode="softmax_topk", device="auto")
# inside a forward hook on layer i's router:
moescope.route_and_log(router_logits, k=8, layer_id=i, token_offset=step * tokens_per_step)
# at the end of the run:
log = moescope.drain()   # dtype [('token_id','<u4'),('layer_id','<u2'),('expert_id','<u2'),('gate','<f4')]
```

When a full buffer drops records, `drain()` warns. `drain_with_stats()` returns `(records, dropped)`.

## Results

Toy OLMoE (real HF architecture, random weights): 4 layers, 64 experts, k=8, 2048 tokens/step (65,536 records/step). Each run takes the median of 10 to 20 steps.

**Apple Silicon CPU** (`bench/results/cpu-apple.json`; moescope CPU backend):

| strategy | step ms | overhead vs none | finalize ms |
|---|---:|---:|---:|
| none | 86.95 | – | – |
| python-list | 93.79 | +6.83 ms (+7.9%), p90 135 ms | 65.4 |
| torch-cpu-copy | 87.68 | +0.72 ms (+0.8%) | 2.2 |
| moescope | 98.49 | +11.54 ms (+13.3%) | 1.9 |

On CPU, moescope **loses**. There is no device sync for it to avoid, and it re-runs routing per token. The CPU backend is a reference and a fallback, not the product.

**T4 GPU:** *not measured yet.* Run `bash scripts/colab.sh` and paste the table it prints (also saved to `bench/results/t4.json`). That comparison is the point of the project: `python-list` and `torch-cpu-copy` force a device→host sync per layer per step, while moescope doesn't. `torch-gpu-list` also avoids syncs and is the honest competitor. Its memory grows without a bound, whereas moescope has a fixed capacity with explicit drop counting.

## Design notes

- **Ties / NaN:** higher logit wins; on an exact tie the lower expert index wins; NaN ranks as -inf. Host APIs reject non-finite logits with an error and never panic. The device-pointer path can't check for NaN.
- **Overflow:** each token reserves k slots with one `atomicAdd` on a 64-bit cursor. Slots at or beyond capacity are not written and are counted in `dropped`.
- **Zero-copy from torch:** the pointer is validated with `cuMemGetAddressRange` and the device ordinal before launch, and the kernel runs on the legacy default stream (torch's default). Validation works per *driver allocation*: torch pools tensors, so a wrong length could read a neighbouring tensor, but never unmapped memory.
- **Limits:** `k ≤ 32`, `num_experts ≤ 65536`, all flat indices must fit in u32.
