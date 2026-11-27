<div align="center">

# 🔬 moescope

**See exactly where every token goes inside a Mixture-of-Experts model.**

moescope logs every routing decision on the GPU, inside a CUDA kernel, and hands you the whole log in one call.

![Rust](https://img.shields.io/badge/Rust-core-orange?logo=rust)
![CUDA](https://img.shields.io/badge/CUDA-NVRTC-76B900?logo=nvidia)
[![PyPI](https://img.shields.io/pypi/v/moescope)](https://pypi.org/project/moescope/)
![Python](https://img.shields.io/badge/Python-PyO3-3776AB?logo=python&logoColor=white)
![Overhead](https://img.shields.io/badge/overhead-%3C3%25-brightgreen)
![Parity](https://img.shields.io/badge/router%20parity-100%25-brightgreen)
[![CI](https://github.com/Knightkolla/moescope/actions/workflows/ci.yml/badge.svg)](https://github.com/Knightkolla/moescope/actions/workflows/ci.yml)

</div>

![Routing report for Granite-3.1-1B-A400M](docs/granite_routing_report.png)

<p align="center"><sub>IBM Granite-3.1-1B-A400M on a free Colab T4: 1.63M routing decisions across code, maths, English and French.</sub></p>

---

## Why

MoE models send each token to a few of many experts. To find out which experts get overloaded, which sit idle and which ones specialise, you need **every** routing decision, and logging that from Python is slow: a naive `.cpu().tolist()` hook nearly **doubles** step time.

moescope keeps the whole log on the GPU instead:

```
router logits ──▶ CUDA kernel ──▶ 12-byte records in a GPU buffer ──▶ drain() ──▶ numpy
                  (top-k + gates)  (token, layer, expert, gate)        (one copy, at the end)
```

No host sync per step and no Python lists. Every decision is either logged or counted as dropped, so nothing goes missing silently.

## Results

Four MoE models from three labs, each hooked on every MoE layer and fed the same code, maths, English and French text on Tesla T4s:

| | Granite-3.1-1B-A400M | Granite-3.1-3B-A800M | OLMoE-1B-7B | LFM2-8B-A1B |
|---|---:|---:|---:|---:|
| Lab | IBM | IBM | AllenAI | Liquid AI |
| MoE layers × experts, top-k | 24 × 32, top-8 | 32 × 40, top-8 | 16 × 64, top-8 | 22 × 32, top-4 |
| Routing | top-k → softmax | top-k → softmax | softmax → top-k | sigmoid + selection bias, renormalized |
| Decisions logged | **1,631,232** | **2,174,976** | **1,017,856** | **680,064** |
| Dropped | 0 | 0 | 0 | 0 |
| Match with the model's own router | **100.00%** (203,904) | **100.00%** (271,872) | **100.00%** (127,232) | **100.00%** (170,016) |
| Forward-pass cost of logging | +2.8%¹ | **+0.7%** | **+0.7%** | +2.6%² |
| Busiest expert vs fair share | 1.8–4.0× | 2.3–5.0× | 3.1–4.8× | 1.6–3.9× |
| Near-idle experts (<10% of fair share) | 17 of 768 | 64 of 1,280 | 26 of 1,024 | 3 of 704 |

"Match" compares moescope's expert set for every (layer, token) with the experts the model itself picked in the same forward pass.

The same pattern shows up in the first three: in the most specialised layer, **code and maths share an expert** (Granite-1B layer 14: expert 13; Granite-3B layer 22: expert 27; OLMoE layer 15: experts 17 and 30), while prose and French spread across others. In OLMoE, routing gets both more domain-specific and more confident with depth.

<table>
<tr>
<td><img src="docs/granite3b_routing_report.png" alt="Routing report for Granite-3.1-3B-A800M"></td>
<td><img src="docs/olmoe_routing_report.png" alt="Routing report for OLMoE-1B-7B"></td>
</tr>
<tr>
<td align="center"><sub>Granite-3.1-3B-A800M · 2.17M decisions</sub></td>
<td align="center"><sub>OLMoE-1B-7B · 1.02M decisions</sub></td>
</tr>
</table>

> Each run uses ~8k tokens of hand-written text. The numbers show what the log can reveal; they are not a study of these models (see [Routing at scale](#routing-at-scale)). Granite-1B ran on Colab; the others on Kaggle (transformers 5.20).
>
> ¹ Measured with the earlier one-thread-per-token kernel. ² fp16 model split over two T4s; the logits of layers on the second GPU are copied to the logging GPU first. Since 0.3 that copy has no host sync, which measured the same (+3.4% in a later run): with ~240-token batches this forward is launch-bound, so the per-layer hook cost is the likely source (not profiled).

## Routing at scale

The results above come from ~8k tokens. `examples/corpus_study.py` repeats the analysis on real corpora: **~100k tokens per domain** (code: MBPP + HumanEval, maths: GSM8K, English: WikiText-103, French: Wikipedia), **51.1M routing decisions** from OLMoE-1B-7B, 0 dropped, logged in 70 s on one T4. Intervals are 95% bootstrap intervals over 512-token chunks.

![Routing study for OLMoE-1B-7B](docs/olmoe_study.png)

1. **Code and maths really do share experts.** Overlap of their 8 most-used experts per layer (Jaccard): **0.37 (0.35–0.38)**. Every other domain pair: 0.05–0.22, with English–French next at 0.22. The peak is layer 8, where code and maths share 7 of their top 8 experts.
2. **Load imbalance is real and stable:** the busiest expert gets **3.55× (3.51–3.59)** its fair share, 2.7–4.1× in every layer.
3. **Idle experts mostly disappear with enough data.** Only **2 of 1,024** experts fall below 10% of a fair share, versus 26 on the 8k-token run. So the near-idle counts in the table above are mainly a small-sample effect.
4. **Position barely matters:** tokens at positions < 64 and ≥ 256 are routed almost identically (JS divergence ≤ 0.011 bits, except 0.035 in layer 0).
5. **Routing drifts during generation, more in deeper layers.** Over 512 greedily generated tokens (8 prompts), the JS divergence between the first and last quarter grows from 0.01 bits in layer 0 to 0.06–0.09 bits in layers 8–15.

> One model, four domains, greedy decoding: a measurement, not a general claim about MoE models. The raw numbers are in [`docs/olmoe_study.json`](docs/olmoe_study.json).

### Same study, DeepSeek-style routing: LFM2-8B-A1B

LFM2-8B-A1B routes like DeepSeek-V3 (sigmoid scores, a per-expert selection bias, renormalized top-k): 22 MoE layers × 32 experts, top-4. Same corpora, **35.0M decisions** over 397k tokens, 0 dropped, model split over two T4s.

| | OLMoE-1B-7B (softmax) | LFM2-8B-A1B (sigmoid + bias) |
|---|---:|---:|
| Pair sharing the most top-8 experts | code–maths **0.37** (0.35–0.38) | English–French **0.48** (0.46–0.51) |
| code–maths overlap | 0.37 | 0.29 (0.27–0.30) |
| Other pairs | 0.05–0.22 | 0.16–0.26 |
| Overlap of two random top-8 sets | 0.07 | 0.14 |
| Busiest expert vs fair share | 3.55× (3.51–3.59) | **2.88×** (2.86–2.89), 1.5–6.4× per layer |
| Near-idle experts (<10% of fair share) | 2 of 1,024 | 6 of 704 |
| Position effect (JS, early vs late positions) | ≤ 0.035 bits | ≤ 0.06 bits |
| Generation drift (JS, first vs last quarter) | 0.01 → 0.06–0.09 bits, grows with depth | 0.016–0.084 bits, mean 0.039, no depth trend |

The bias-balanced router spreads load more evenly on average (2.9× vs 3.6×), but not uniformly: one layer still sends 6.4× the fair share to its busiest expert. Where OLMoE groups code with maths, LFM2 groups the two natural languages. Overlap numbers are not directly comparable across the two models (8 of 32 experts is a quarter of a layer, 8 of 64 an eighth), hence the random baselines.

## Which experts does a workload need?

`examples/expert_budget.py` turns a routing log into a pruning or offload plan and then measures what it costs. On OLMoE-1B-7B it logs ~80k tokens of code, keeps per layer the experts that carry 90 / 95 / 99 / 99.9% of the code routing's gate mass, and measures perplexity on **held-out** code while the router may only pick kept experts (it falls back to its next-best allowed expert). Controls: the same number of experts per layer chosen from a French log, and chosen at random.

| experts kept per layer | chosen from the code log | from a French log | random |
|---:|---:|---:|---:|
| 64 (full model) | ppl 3.644 | | |
| 59.9 | **+0.2%** | +13.7% | +22.1% |
| 51.4 | **+0.2%** | +43.0% | +104% |
| 36.8 | **+1.3%** | +680% | +523% |
| 27.8 | **+2.4%** | +2,551% | +9,549% |

Keeping **43% of the experts** costs code **2.4%** perplexity when they are chosen from the code log, and destroys the model when they are not. For CPU offload, the same log says how much routing a GPU-resident set serves on held-out code:

| experts on GPU per layer | 8 | 16 | 24 | 32 | 48 |
|---|---:|---:|---:|---:|---:|
| busiest on the code log | 54% | **74%** | 85% | **92%** | 98% |
| busiest on a French log | 14% | 26% | 37% | 50% | 75% |
| (no plan: uniform) | 13% | 25% | 38% | 50% | 75% |

> One model, one target domain, held-out text from the same code datasets (MBPP / HumanEval): in-distribution, not a claim about general code. Perplexity is not task accuracy. The helpers are in `moescope.analysis` (`expert_usage`, `experts_for_coverage`, `resident_hit_rate`, `budget_curve`).

## Quick start

```sh
pip install moescope
```

```python
import moescope

with moescope.attach(model) as session:   # any supported Hugging Face MoE model
    model(**inputs)
log = session.drain()
```

On Linux the wheel includes the CUDA backend. It is loaded at run time, works with CUDA 12.4+ and 13 drivers, and falls back to the CPU router when there is no NVIDIA driver. macOS wheels are CPU only.

<details>
<summary>Build from source</summary>

```sh
pip install maturin
maturin build --release -F python,pyo3/extension-module,cuda-12040 -o dist   # or cuda-13000 / cuda-auto
pip install dist/moescope-*.whl
```

CPU only: `maturin develop --release`. You need a Rust toolchain; the GPU path also needs NVRTC (from the CUDA toolkit) at run time.

</details>

## Usage

```python
import moescope

with moescope.attach(model) as session:      # finds every MoE router, reads the routing rule from the config
    model.generate(**inputs)                 # or training steps, or plain forward passes

log = session.drain()                        # structured numpy array, one row per decision
log["token_id"], log["layer_id"], log["expert_id"], log["gate"], log["flags"]
```

`attach` hooks each router and logs its own logits: one kernel launch per layer, no recomputation, no host sync. `token_id` counts tokens in processing order, so it keeps working across `generate` steps (in encoder-decoder models, encoder and decoder passes share the counter, and decoder layer ids follow the encoder's).

**Checking the log against what was computed.** With `check=True`, `session.parity(log)` compares the log with what the experts *received*, not with what the router returned: the `top_k_index` each transformers MoE block hands its experts, or, for models with one module per expert, the rows that actually reached each expert module. A (layer, token) matches when the experts that ran on it are exactly its logged, non-dropped decisions.

**Capacity drops.** Switch Transformer and NLLB-MoE give each expert a fixed capacity, and tokens past it are not computed by that expert. `attach` passes the router's own dispatch mask to the kernel, so those decisions are still logged (with the router's gate) and carry `flags & moescope.CAPACITY_DROP`. On Switch Transformer the log matches the experts' actual inputs exactly, drops included.

Supported routing. Families in **bold** are tested against the real transformers code in `tests/python/test_hf_attach.py` (every expert choice identical, gates within 1e-7); the others use routing code identical to a tested family:

| family | routing | moescope mode |
|---|---|---|
| **OLMoE**, **Qwen2/3-MoE**, **DeepSeek-V2** (`greedy`) | softmax over all experts → top-k (± renormalize, × scale) | `softmax_topk` / `topk_softmax` |
| **Mixtral**, **Granite**, MiniMax, HunYuan | top-k logits → softmax over the k | `topk_softmax` |
| **DeepSeek-V3**, **GLM-4.5**, **LFM2-MoE**, Kimi K2, dots1 | sigmoid + per-expert selection bias, group-limited top-k, renormalize, × scale | `sigmoid` with bias and groups |
| **Switch Transformer** | softmax → top-1, per-sequence expert capacity | `softmax_topk`, drops flagged |
| NLLB-MoE (see below) | top-2 → softmax over the two, expert capacity | `topk_softmax`, drops flagged |
| Llama 4 (not yet tested: its router doesn't expose the chosen experts) | top-k logits → sigmoid | `sigmoid` |

NLLB-MoE in transformers 5.x does not compute what its router decides: `NllbMoeExperts` one-hot encodes the router's mask a second time, so experts 0 and 1 receive every token and the chosen experts never run (on a tiny model: router picks experts {1, 2, 3}; expert 0 gets 30 rows from 10 tokens, experts 2 and 3 get none). moescope logs the router's decision; `parity()` reports 0% agreement with the experts. `tests/python/test_hf_attach.py` keeps this as a regression test.

`attach` refuses what it can't reproduce exactly (DeepSeek-V2 `group_limited_greedy`, ERNIE-4.5) instead of logging something subtly wrong.

<details>
<summary>Lower-level API (no Hugging Face)</summary>

```python
moescope.configure(capacity=50_000_000, mode="sigmoid", normalize=True, scale=2.5, n_group=8, topk_group=4)
moescope.set_bias(layer_id=3, bias=router.e_score_correction_bias)      # once per layer
moescope.route_and_log(logits, k=8, layer_id=3, token_offset=offset)    # each step
log = moescope.drain()
```

- **Zero-copy:** a torch CUDA tensor is read in place: one pointer check, one kernel launch.
- **Overflow:** if the buffer fills up, `drain()` warns and `drain_with_stats()` returns the dropped count.

</details>

A full report (figure, raw log, `summary.json`):

```sh
python examples/analyze_routing.py                                       # Granite-3.1-1B-A400M
python examples/analyze_routing.py --model allenai/OLMoE-1B-7B-0924
python examples/analyze_routing.py --model LiquidAI/LFM2-8B-A1B --device-map auto
python examples/analyze_routing.py --toy deepseek_v3                     # tiny offline model
```

## Serving and CUDA graphs

- **No host sync.** On the GPU, logging launches on the caller's current CUDA stream (`torch.cuda.current_stream()`), so it is ordered after the router that produced the logits without any synchronization, even on non-blocking streams.
- **Half precision in place.** float16 and bfloat16 logits are read directly; nothing is converted or copied. For routers that compute `sigmoid` in half precision (LFM2-MoE), moescope rounds the scores the same way torch does, so the logged choice stays identical.
- **CUDA graphs.** Logging can be captured into a CUDA graph and replayed. Token ids come from a device-side counter that a tiny kernel advances after each forward pass, so every replay logs fresh ids. `attach()` does this automatically on the GPU; with the low-level API, call `recorder.set_device_token_ids(True)` and `moescope.advance_tokens(n)` once per pass.

- **Models split over GPUs** (`device_map`): logits from another GPU are copied to the log's GPU in stream order, with no host sync.

### vLLM

```python
from vllm import LLM
import moescope.vllm

llm = LLM("ibm-granite/granite-3.1-1b-a400m-instruct", enforce_eager=True)
session = moescope.vllm.attach(llm)          # hooks run in vLLM's worker process
llm.generate(prompts)
log = session.drain()
```

Tested with vLLM 0.34 on a T4 (`scripts/vllm_check.sh`, Granite-3.1-1B-A400M): every processed token logged once per layer × top-8 (25,536 records, 0 lost), and on the same prompt vLLM and transformers pick the same expert set for 99.55% of (layer, token) pairs (both fp16, different kernels).

The hook sits on every MoE layer's `forward(hidden_states, router_logits)`, the interface all vLLM MoE layers share, so it does not depend on vLLM's internal kernel classes. With tensor parallelism only rank 0 logs (router logits are replicated). Limits: it needs `enforce_eager=True` (vLLM compiles the model with `fullgraph=True`, where Python hooks cannot run), `token_id` counts processed tokens across the whole batch rather than per request, and prefix-cache hits are not recomputed, so they are not logged. vLLM may also need `VLLM_ALLOW_INSECURE_SERIALIZATION=1` to send the hook function to its worker.

## Training

```python
from moescope.monitor import MoescopeCallback
trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[MoescopeCallback(every=100)])
```

Every `every` steps it reports, per MoE layer and averaged: load imbalance (busiest expert / fair share), dead experts, load entropy, capacity-drop rate, and drift (JS divergence of the expert load from the previous window). Metrics go to Weights & Biases when a run is active, or to any `log_fn(stats, step=...)`. Without the Trainer, wrap a session in `moescope.monitor.RoutingMonitor(session, every=100, log_fn=...)` and call `monitor.step()` after each optimizer step. The log is drained into counts every step, so the buffer only has to hold one step. With gradient checkpointing, the recomputed forward in backward is not logged a second time.

## Benchmark

A toy OLMoE (4 layers, 64 experts, top-8, 2,048 tokens per step) on a T4, logging every decision:

| strategy | step (ms) | overhead |
|---|---:|---:|
| no logging | 23.39 | – |
| Python lists, `.cpu().tolist()` | 42.84 | **+83.2%** |
| `.cpu()` tensors each step | 24.08 | +3.0% |
| tensors kept on GPU, `torch.cat` at end | 22.43 | within noise |
| **moescope** | **22.70** | **within noise** |

moescope costs about the same as keeping tensors on the GPU yourself. What it adds is a bounded, pre-allocated buffer, explicit drop counting, a fixed record format and a ready-made analysis report. Reproduce with `bench/bench_hook.py`.

## Stress test

`bench/stress_test.py` pushes the logging path far beyond a normal forward pass and checks every result against torch. All checks pass on a T4:

| check | result |
|---|---|
| Correctness | 66 (experts, k, mode) combinations, E = 8 to 256, k = 1 to 32: **17.3M tokens**, every expert choice matches torch, worst gate error 9e-7 |
| Volume | **98.6M decisions (1.18 GB)** in one buffer, 0 dropped, per-expert histogram identical to torch's, drained in 4.9 s |
| Overflow | 3M decisions into a 1M-record buffer: kept exactly 1,000,000, counted exactly 2,000,000 dropped; buffer reusable after drain; 200 log/drain cycles |
| Odd inputs | all-equal logits, NaN, ±1e30, k = E, E = 65,536, fp16 / bf16, non-contiguous and 3-D tensors, logits produced on a side CUDA stream |

Throughput of one `route_and_log` call (T4, 1M tokens; torch only computes the routing, moescope also logs it):

| routing | moescope | torch, same routing |
|---|---:|---:|
| 8 experts, top-2 | **0.34 ms** (6.2 G decisions/s) | 17.5 ms |
| 32 experts, top-8 | **3.7 ms** | 18.2 ms |
| 64 experts, top-8 | **8.0 ms** | 26.0 ms |
| 128 experts, top-8 | **16.5 ms** | 43.8 ms |
| DeepSeek-V3: 256 experts, sigmoid + bias, 8 groups / top 4, top-8 | **50.9 ms** | 268 ms |
| 64 experts, fp16 / bf16 logits read in place | **8.1 / 8.9 ms** | – |

Each token is handled by a group of 2–32 threads, sized to the expert count. They read the row together in one coalesced pass, pick the top-k in k rounds of warp-shuffle reductions, and keep only their own ≤ 4 output slots, in registers (no local memory). The kernel is compiled once per (logits dtype, softmax/sigmoid) combination, so neither choice costs anything in the inner loop.

## What it is / isn't

| It is | It isn't |
|---|---|
| A low-overhead recorder for load imbalance, dead experts and specialisation | A faster MoE router: the model's forward pass is unchanged |
| GPU kernels tested against a CPU reference (exact ids, gates within 1e-5) | Bit-identical to torch on exact ties (the lower expert index wins) |
| A fixed-capacity log that never grows without bound or silently loses data | |

## How it works

| path | role |
|---|---|
| `src/router.rs` | CPU reference router: softmax / sigmoid gating, selection bias, group-limited top-k, NaN-safe validation |
| `src/dispatcher.rs` | groups tokens per expert, builds the load histogram |
| `src/telemetry.rs` | the 12-byte `#[repr(C)]` record and a bounded host log |
| `src/gpu/kernels.rs` | CUDA C compiled at runtime with NVRTC: a lane group per token (warp shuffles), fused route + dispatch + log |
| `src/gpu/raw.rs` | the **only** module allowed `unsafe`; every block has a safety comment |
| `src/python.rs`, `python/moescope/` | PyO3 bindings, the numpy / torch API, `attach()` for Hugging Face models (`hf.py`), vLLM (`vllm.py`), training metrics (`monitor.py`), usage / pruning / offload analysis (`analysis.py`) |

<details>
<summary>Design notes</summary>

- **Ties and NaN:** the higher logit wins. On an exact tie the lower expert index wins. NaN ranks below every number.
- **Appending:** each token reserves its k log slots with one 64-bit `atomicAdd`. Records past capacity are counted as `dropped`, not written.
- **Order:** a token's k records are contiguous and in rank order (best expert first), but the order of tokens follows which thread group won the atomic, so it changes between runs. `drain(sort=True)` (or `moescope.sort_records`) gives a canonical layer → token → rank order, identical byte for byte across runs on the same inputs. The analysis helpers don't depend on order.
- **Foreign pointers:** a torch tensor's address is checked with `cuMemGetAddressRange` and its device ordinal is verified before the kernel reads it. The check works per driver allocation: torch pools tensors, so a wrong length could read a neighbouring tensor, but never unmapped memory.
- **Record:** 12 bytes, `token_id u32, layer_id u8, flags u8, expert_id u16, gate f32` (0.3 split the old u16 `layer_id` into `layer_id` + `flags`).
- **Limits:** `k ≤ 32`, `num_experts ≤ 65,536`, layer ids ≤ 255, and every flat index must fit in u32.

</details>

## Testing

```sh
cargo test                                # CPU: router, dispatch, telemetry, property tests
cargo test --features cuda-13000          # builds the GPU path (GPU tests skip without a driver)
python -m pytest -q tests/python          # bindings, attach() parity on 14 model configurations, vLLM hooks, monitor
bash scripts/colab.sh cuda-13000          # full GPU suite, ends with "ALL PHASES PASSED"
bash scripts/colab_models.sh              # GPU parity + stream/graph tests, stress test, routing reports
```

On a GPU machine, set `MOESCOPE_REQUIRE_GPU=1` so a missing driver fails the tests instead of skipping them.

## License

MIT, see [LICENSE](LICENSE).
