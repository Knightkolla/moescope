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

**Colab Tesla T4** (torch 2.14.1+cu130, driver CUDA 13.0, fp16 model, moescope CUDA backend, 20 timed steps; produced by `scripts/colab.sh`):

| strategy | step ms (median) | p90 ms | overhead vs none | finalize ms | records |
|---|---:|---:|---:|---:|---:|
| none | 23.39 | 28.65 | – | – | 0 |
| python-list | 42.84 | 66.46 | +19.45 ms (+83.2%) | 446.0 | 1,507,328 |
| torch-cpu-copy | 24.08 | 25.25 | +0.69 ms (+3.0%) | 38.2 | 1,507,328 |
| torch-gpu-list | 22.43 | 23.57 | −0.96 ms (−4.1%) | 46.9 | 1,507,328 |
| moescope | 22.70 | 23.47 | −0.68 ms (−2.9%) | 46.8 | 1,507,328 |

How to read this:
- Naive per-step Python logging nearly doubles step time and adds a 0.45 s finalize.
- moescope's overhead is **within run-to-run noise** of no logging at all. The `none` row ran first and has a wide p90, so the small negative numbers mean "no measurable cost", not "faster".
- moescope is no faster than `torch-gpu-list`, which also avoids per-step syncs. What moescope adds over that is a fixed, pre-allocated buffer with explicit drop counting (the list grows without bound) and a ready-made structured record format.
- `torch-cpu-copy` costs only about 3% here, because this toy model's per-layer sync is cheap. Expect the gap to grow with deeper models and asynchronous pipelines.

Parity on the same T4: GPU router tests (exact expert ids, gates within 1e-5) pass. Through the zero-copy CUDA path, the HF hook example matches the model's own router on every token, with max gate diff 2.2e-8 (OLMoE) and 1.2e-7 (Mixtral).

## Real model: Granite-3.1-1B-A400M on a T4

`python examples/analyze_routing.py` hooks all 24 MoE layers of IBM's Granite-3.1-1B-A400M (32 experts, top-8) and feeds it short code, maths, English prose and French passages. Results on a Colab T4:

- **1,631,232 routing decisions** logged (8,496 tokens × 24 layers × top-8), 0 dropped, drained in one call.
- **100.00% parity**: 203,904 of 203,904 (layer, token) expert sets match the model's own router.
- **Logging cost: +2.8%** forward time (366.7 → 377.1 ms). An earlier build cost +13.9%: it allocated dispatch buffers on every call and recomputed the router matmul. Logging now uses a telemetry-only kernel (one pointer check plus one launch, no allocations) on the router's own logits.
- **Load imbalance:** the busiest expert gets 1.8–4.0× its fair share depending on layer. 17 experts receive under 10% of a fair share on this text.
- **Specialisation:** layer 14 separates domains the most. Code and maths both lean on the same expert there, while prose and French spread across others.

The script saves `routing_report.png`, the raw log (`routing_log.npz`) and `summary.json`. It works with OLMoE and Mixtral too (`--model ...`), and `--toy granitemoe|olmoe|mixtral` runs it offline on a tiny random model.

These findings come from a few thousand tokens of hand-written text. They are a demo of what the log shows, not a study of Granite.

## Design notes

- **Ties / NaN:** higher logit wins; on an exact tie the lower expert index wins; NaN ranks as -inf. Host APIs reject non-finite logits with an error and never panic. The device-pointer path can't check for NaN.
- **Overflow:** each token reserves k slots with one `atomicAdd` on a 64-bit cursor. Slots at or beyond capacity are not written and are counted in `dropped`.
- **Zero-copy from torch:** the pointer is validated with `cuMemGetAddressRange` and the device ordinal before launch, and the kernel runs on the legacy default stream (torch's default). Validation works per *driver allocation*: torch pools tensors, so a wrong length could read a neighbouring tensor, but never unmapped memory.
- **Limits:** `k ≤ 32`, `num_experts ≤ 65536`, all flat indices must fit in u32.
