# moescope

**Log every routing decision of a Mixture-of-Experts model on the GPU, then read it all back in one call.**

moescope is a small Rust + CUDA library with Python bindings. You attach it to an MoE model's router. A CUDA kernel recomputes each token's top-k experts and gate weights, and appends one 12-byte record per decision, `(token_id, layer_id, expert_id, gate)`, to a fixed-size buffer in GPU memory. Nothing crosses to the host until you call `drain()`, which returns the whole log as a numpy array.

![Routing report for Granite-3.1-1B-A400M](docs/granite_routing_report.png)

*Granite-3.1-1B-A400M (24 MoE layers, 32 experts, top-8) on a Colab T4: 1.63M routing decisions over code, maths, English and French text. Top: load per expert and layer. Bottom: per-domain expert usage in the most specialised layer.*

## Results

Measured on a Colab Tesla T4 with `examples/analyze_routing.py`:

| | |
|---|---|
| Decisions logged | 1,631,232 (8,496 tokens × 24 layers × top-8), 0 dropped |
| Match with the model's own router | 100.00% (203,904 of 203,904 expert sets) |
| Forward-pass cost of logging | +2.8% (366.7 → 377.1 ms) |
| Load imbalance | busiest expert gets 1.8–4.0× its fair share, depending on layer |
| Near-idle experts | 17 across all layers (under 10% of a fair share) |
| Most specialised layer | 14: code and maths both lean on expert 13, prose and French use others |

These numbers come from a few thousand tokens of hand-written text. They show what the log can reveal; they are not a study of Granite.

## What it is, and what it isn't

**It is**
- A low-overhead way to record every routing decision, for spotting load imbalance, dead experts and domain specialisation.
- A CPU reference router. The CUDA kernels are tested against it: exact expert ids, gates within 1e-5.
- A fixed-capacity log with explicit drop counting. It never grows without bound and never loses records silently.

**It isn't**
- A faster MoE router. It recomputes routing next to the model's own router and doesn't change the forward pass.
- Bit-identical to torch on exact ties: ties go to the lower expert index.

## Install

GPU (Linux + NVIDIA). Pick the `cuda-XXXXX` feature that matches the "CUDA Version" in `nvidia-smi`; supported values are `cuda-13000`, `cuda-12080`, `cuda-12040` and `cuda-auto`:

```sh
pip install maturin
maturin build --release -F python,pyo3/extension-module,cuda-13000 -o dist
pip install dist/moescope-*.whl
```

CPU only (any OS, for trying it out):

```sh
pip install maturin && maturin develop --release
```

Building needs a Rust toolchain. The GPU build needs the NVIDIA driver and NVRTC (part of the CUDA toolkit) at runtime.

## Usage

```python
import moescope

moescope.configure(capacity=50_000_000, mode="topk_softmax")  # 12 bytes per decision

def make_hook(layer_id):
    def hook(_module, _inputs, out):
        logits = out[0] if isinstance(out, tuple) else out
        moescope.route_and_log(logits, k=8, layer_id=layer_id, token_offset=offset)
    return hook

for i, layer in enumerate(model.model.layers):
    layer.block_sparse_moe.router.register_forward_hook(make_hook(i))  # OLMoE/Mixtral: layer.mlp.gate

# ... run your model as usual ...

log = moescope.drain()  # numpy structured array: token_id, layer_id, expert_id, gate
```

- **Modes:** `softmax_topk` (softmax over all experts, then keep the top-k probabilities) for OLMoE and Switch. `topk_softmax` (softmax over the k selected logits) for Mixtral and Granite.
- **Zero-copy:** a torch CUDA tensor is read in place. Logging is one pointer check plus one kernel launch, with no host sync.
- **Overflow:** if the buffer fills up, `drain()` warns and `drain_with_stats()` returns the dropped count.

For a complete report (figure, raw log, `summary.json`):

```sh
python examples/analyze_routing.py                                # Granite-3.1-1B-A400M
python examples/analyze_routing.py --model allenai/OLMoE-1B-7B-0924
python examples/analyze_routing.py --toy granitemoe               # tiny offline model
```

## Testing

```sh
cargo test                                   # CPU: router, dispatch, telemetry, property tests
cargo test --features cuda-13000             # also builds the GPU path (GPU tests skip without a driver)
python -m pytest -q tests/python             # Python bindings
bash scripts/colab.sh cuda-13000             # everything on a GPU, ending in "ALL PHASES PASSED"
```

On a GPU machine, set `MOESCOPE_REQUIRE_GPU=1` so a missing driver fails the GPU tests instead of skipping them.

## Benchmark: logging strategies

`bench/bench_hook.py` runs a toy OLMoE (4 layers, 64 experts, top-8, 2,048 tokens per step) on a T4 and compares ways of logging every decision:

| strategy | step ms (median) | overhead vs no logging | finalize ms |
|---|---:|---:|---:|
| none | 23.39 | – | – |
| Python lists, `.cpu().tolist()` per step | 42.84 | +83.2% | 446.0 |
| `.cpu()` tensors per step | 24.08 | +3.0% | 38.2 |
| tensors kept on GPU, `torch.cat` at end | 22.43 | within noise | 46.9 |
| moescope | 22.70 | within noise | 46.8 |

Naive Python logging nearly doubles step time. moescope costs about the same as keeping tensors on the GPU yourself. What it adds over that is a bounded, pre-allocated buffer with explicit drop counting and a ready-made record format.

## How it works

| path | role |
|---|---|
| `src/router.rs` | CPU reference: softmax, top-k, both gate modes, `Result`-based validation (no panics on NaN) |
| `src/dispatcher.rs` | groups routed tokens per expert and builds the load histogram |
| `src/telemetry.rs` | the 12-byte `#[repr(C)]` record and a bounded host-side log |
| `src/gpu/kernels.rs` | CUDA C compiled at runtime with NVRTC: one thread per token, fused route + dispatch + log |
| `src/gpu/raw.rs` | the only module allowed `unsafe` (the crate is `#![deny(unsafe_code)]`); every block has a safety comment |
| `src/python.rs`, `python/moescope/` | PyO3 bindings and the numpy/torch-facing API |

Design notes:
- **Ties and NaN:** a higher logit wins. On an exact tie the lower expert index wins. NaN ranks below every number.
- **Appending:** each token reserves its k log slots with one `atomicAdd` on a 64-bit cursor. Records past capacity are counted in `dropped`, not written.
- **Foreign pointers:** a torch tensor's address is checked with `cuMemGetAddressRange` and the device ordinal before the kernel reads it. The check is per driver allocation: torch pools tensors, so a wrong length could read a neighbouring tensor, but never unmapped memory.
- **Limits:** `k ≤ 32`, `num_experts ≤ 65,536`, and every flat index must fit in u32.
