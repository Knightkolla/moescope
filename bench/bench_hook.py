"""Overhead of logging every MoE routing decision during inference.

Runs a toy OLMoE (real HF architecture, random weights) with a forward hook
on every layer's router, and times N forward steps per logging strategy:

  none            no hook: the floor
  python-list     per-step Python logging: .cpu().tolist() into Python lists
                  (what a quick research hook usually looks like)
  torch-cpu-copy  per-step .cpu() of the index/gate tensors, no Python loop
  torch-gpu-list  keep index/gate tensors on device, torch.cat at the end
                  (no per-step sync; memory grows with the run, no cap)
  moescope        moescope.route_and_log (fused kernel writes the log in
                  device memory, zero-copy from the CUDA tensor), drain once

Each strategy must end with the same number of records; the "finalize"
column is the one-time cost of producing a host-side table at the end.

    python bench/bench_hook.py                 # CUDA if available, else CPU
    python bench/bench_hook.py --device cpu --steps 10
"""

import argparse
import json
import os
import platform
import time

import numpy as np
import torch

import moescope

DTYPE = moescope.RECORD_DTYPE


def build_model(args):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(0)
    cfg = OlmoeConfig(vocab_size=1000, hidden_size=args.hidden, intermediate_size=args.hidden // 2,
                      num_hidden_layers=args.layers, num_attention_heads=8, num_key_value_heads=8,
                      num_experts=args.experts, num_experts_per_tok=args.k, max_position_embeddings=args.seq,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    return OlmoeForCausalLM(cfg).to(args.device, dtype).eval()


def unpack(output, k):
    """(logits, indices, gates) from an OLMoE gate module's output (HF 4.x or 5.x)."""
    if isinstance(output, tuple):
        return output[0], output[2], output[1]
    probs = torch.softmax(output.float(), dim=-1)
    gates, idx = torch.topk(probs, k, dim=-1)
    return output, idx, gates


class Strategy:
    """A logging strategy: a per-layer hook plus a one-time finalize()."""

    name = "none"

    def __init__(self, k, tokens_per_step):
        self.k, self.tps, self.step = k, tokens_per_step, 0

    def hook(self, layer_id):
        return None

    def finalize(self):
        return np.zeros(0, DTYPE)


class PythonList(Strategy):
    name = "python-list"

    def __init__(self, *a):
        super().__init__(*a)
        self.rows = []

    def hook(self, layer_id):
        def fn(_m, _i, out):
            _, idx, gates = unpack(out, self.k)
            base = self.step * self.tps
            for t, (es, gs) in enumerate(zip(idx.tolist(), gates.float().tolist())):
                for e, g in zip(es, gs):
                    self.rows.append((base + t, layer_id, e, g))
        return fn

    def finalize(self):
        return np.array(self.rows, dtype=DTYPE)


class TorchCpuCopy(Strategy):
    name = "torch-cpu-copy"

    def __init__(self, *a):
        super().__init__(*a)
        self.chunks = []

    def hook(self, layer_id):
        def fn(_m, _i, out):
            _, idx, gates = unpack(out, self.k)
            self.chunks.append((self.step, layer_id, idx.cpu(), gates.float().cpu()))
        return fn

    def finalize(self):
        return to_records(self.chunks, self.k, self.tps)


class TorchGpuList(Strategy):
    name = "torch-gpu-list"

    def __init__(self, *a):
        super().__init__(*a)
        self.chunks = []

    def hook(self, layer_id):
        def fn(_m, _i, out):
            _, idx, gates = unpack(out, self.k)
            self.chunks.append((self.step, layer_id, idx.clone(), gates.float().clone()))
        return fn

    def finalize(self):
        return to_records([(s, l, i.cpu(), g.cpu()) for s, l, i, g in self.chunks], self.k, self.tps)


class Moescope(Strategy):
    name = "moescope"

    def __init__(self, k, tps, capacity, device):
        super().__init__(k, tps)
        self.rec = moescope.Recorder(capacity=capacity, mode="softmax_topk", device=device)

    def hook(self, layer_id):
        def fn(_m, _i, out):
            logits, _, _ = unpack(out, self.k)
            moescope.route_and_log(logits, self.k, layer_id, self.step * self.tps, recorder=self.rec)
        return fn

    def finalize(self):
        recs, dropped = moescope.drain_with_stats(self.rec)
        assert dropped == 0, dropped
        return recs


def to_records(chunks, k, tps):
    parts = []
    for step, layer, idx, gates in chunks:
        n = idx.shape[0]
        r = np.empty(n * k, DTYPE)
        r["token_id"] = np.repeat(np.arange(n) + step * tps, k)
        r["layer_id"] = layer
        r["expert_id"] = idx.reshape(-1).numpy()
        r["gate"] = gates.reshape(-1).numpy()
        parts.append(r)
    return np.concatenate(parts) if parts else np.zeros(0, DTYPE)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run(strategy, model, args):
    gates = [layer.mlp.gate for layer in model.model.layers]
    handles = [g.register_forward_hook(h) for i, g in enumerate(gates) if (h := strategy.hook(i))]
    ids = torch.randint(0, 1000, (args.batch, args.seq), device=args.device)
    times = []
    with torch.no_grad():
        for step in range(args.warmup + args.steps):
            strategy.step = step
            sync(args.device)
            t0 = time.perf_counter()
            model(input_ids=ids)
            sync(args.device)
            if step >= args.warmup:
                times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    records = strategy.finalize()
    finalize_s = time.perf_counter() - t0
    for h in handles:
        h.remove()
    return np.array(times), finalize_s, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tps = args.batch * args.seq
    total_records = (args.warmup + args.steps) * tps * args.k * args.layers
    mbackend = "cuda" if args.device == "cuda" and moescope.cuda_available() else "cpu"
    model = build_model(args)
    strategies = [
        Strategy(args.k, tps),
        PythonList(args.k, tps),
        TorchCpuCopy(args.k, tps),
        TorchGpuList(args.k, tps) if args.device == "cuda" else None,
        Moescope(args.k, tps, total_records, mbackend),
    ]
    env = {
        "device": torch.cuda.get_device_name() if args.device == "cuda" else f"CPU ({platform.platform()})",
        "torch": torch.__version__,
        "moescope_backend": mbackend,
        "tokens_per_step": tps,
        "records_per_step": tps * args.k * args.layers,
        **{k: getattr(args, k) for k in ("layers", "experts", "k", "steps")},
    }
    print(json.dumps(env))
    rows = []
    base = None
    for s in filter(None, strategies):
        times, fin, recs = run(s, model, args)
        med = float(np.median(times)) * 1e3
        base = med if base is None else base
        expected = 0 if s.name == "none" else total_records
        assert len(recs) == expected, (s.name, len(recs), expected)
        rows.append(dict(strategy=s.name, step_ms=med, p90_ms=float(np.percentile(times, 90)) * 1e3,
                         overhead_ms=med - base, overhead_pct=100 * (med - base) / base,
                         finalize_ms=fin * 1e3, records=len(recs)))

    print(f"\n| strategy | step ms (median) | p90 ms | overhead vs none | finalize ms | records |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['strategy']} | {r['step_ms']:.2f} | {r['p90_ms']:.2f} | "
              f"{r['overhead_ms']:+.2f} ms ({r['overhead_pct']:+.1f}%) | {r['finalize_ms']:.1f} | {r['records']:,} |")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"env": env, "results": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
