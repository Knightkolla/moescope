"""GPU stress test for moescope's logging path.

Checks, against a torch reference, at sizes far beyond a normal forward pass:

  1. correctness   every (E, k) combo, 262k tokens: same experts, same order, gates within 1e-5
  2. volume        100M decisions in one buffer: exact count, per-expert histogram == torch's
  3. overflow      3x more decisions than capacity: exact drop count, buffer reusable after drain
  4. adversarial   ties, NaN, huge magnitudes, k = E, k = 32, fp16/bf16, non-contiguous, side stream
  5. throughput    decisions/s for one route_and_log call, vs torch.topk alone

    python bench/stress_test.py --out bench/results/stress_t4.json

Exit code 0 and "ALL STRESS TESTS PASSED" mean every check held.
"""

import argparse
import json
import time

import numpy as np
import torch

import moescope

DEV = "cuda"
results = {"checks": {}, "throughput": []}
failures = []


def check(name, ok, detail=""):
    results["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def reference(logits, k, mode):
    """torch's routing decision: (ids [N,k], gates [N,k]), best first."""
    x = logits.float()
    if mode == "softmax_topk":
        gates, ids = torch.softmax(x, -1).topk(k, -1)
    else:
        vals, ids = x.topk(k, -1)
        gates = torch.softmax(vals, -1)
    return ids, gates


def agree(logits, ids, k, mode):
    """(rows with identical ids, rows whose chosen experts have identical
    ranking values to torch's). The second differs from the first only on
    exact fp ties, where torch's pick is arbitrary and both are correct."""
    x = logits.reshape(-1, logits.shape[-1]).float()
    q = torch.softmax(x, -1) if mode == "softmax_topk" else x
    rids, _ = reference(x, k, mode)
    mine = torch.from_numpy(ids).to(x.device)
    exact = float((mine == rids).all(1).float().mean())
    by_value = float((q.gather(1, mine) == q.gather(1, rids)).all(1).float().mean())
    return exact, by_value


def log_once(logits, k, mode, capacity=None, layer_id=0, token_offset=0):
    n = logits.reshape(-1, logits.shape[-1]).shape[0]
    moescope.configure(capacity=capacity or n * k, mode=mode, device="cuda")
    moescope.route_and_log(logits, k=k, layer_id=layer_id, token_offset=token_offset)
    return moescope.drain_with_stats()


def per_token(log, n, k):
    """Records grouped by token, in the kernel's best-first order."""
    srt = log[np.argsort(log["token_id"], kind="stable")]
    return srt["expert_id"].reshape(n, k).astype(np.int64), srt["gate"].reshape(n, k)


def test_correctness():
    print("1) correctness vs torch, 262,144 tokens per case")
    n = 1 << 18
    g = torch.Generator(device=DEV).manual_seed(0)
    cases = []
    for e in (8, 16, 32, 40, 64, 128, 256):
        for k in (1, 2, 4, 8, 32):
            if k > e:
                continue
            for mode in ("softmax_topk", "topk_softmax"):
                logits = torch.randn(n, e, device=DEV, generator=g) * 3
                log, dropped = log_once(logits, k, mode)
                ids, gates = per_token(log, n, k)
                _, rgates = reference(logits, k, mode)
                exact, by_value = agree(logits, ids, k, mode)
                gate_err = float(np.abs(gates - rgates.cpu().numpy()).max())
                ok = dropped == 0 and by_value == 1.0 and gate_err < 1e-5
                cases.append(dict(experts=e, k=k, mode=mode, exact_id_rows=exact, equal_value_rows=by_value,
                                  max_gate_err=gate_err, ok=ok))
                if not ok:
                    print(f"     mismatch E={e} k={k} {mode}: by value {by_value:.6f}, gate err {gate_err:.2e}")
    n_ok = sum(c["ok"] for c in cases)
    worst = max(c["max_gate_err"] for c in cases)
    ties = sum(round((1 - c["exact_id_rows"]) * n) for c in cases)
    check("correctness", n_ok == len(cases),
          f"{n_ok}/{len(cases)} (E,k,mode) cases, {len(cases) * n:,} tokens, worst gate err {worst:.1e}, "
          f"{ties} rows differ only by an exact tie")
    results["correctness_cases"] = cases


def test_volume(total_decisions):
    print(f"2) volume: {total_decisions:,} decisions in one buffer")
    e, k, chunk = 64, 8, 1 << 18
    calls = total_decisions // (chunk * k)
    moescope.configure(capacity=calls * chunk * k, mode="softmax_topk", device="cuda")
    g = torch.Generator(device=DEV).manual_seed(1)
    ref_hist = torch.zeros(e, dtype=torch.int64, device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for c in range(calls):
        # A random permutation per row: no two logits in a row are equal, so
        # the histogram has exactly one right answer.
        logits = torch.rand(chunk, e, device=DEV, generator=g).argsort(-1).float() * 0.1
        moescope.route_and_log(logits, k=k, layer_id=c % 64, token_offset=c * chunk)
        ref_hist += torch.bincount(logits.topk(k, -1).indices.flatten(), minlength=e)
    torch.cuda.synchronize()
    t_log = time.perf_counter() - t0
    t0 = time.perf_counter()
    log, dropped = moescope.drain_with_stats()
    t_drain = time.perf_counter() - t0
    hist = np.bincount(log["expert_id"], minlength=e)
    tokens_ok = np.array_equal(np.bincount(log["token_id"] // chunk, minlength=calls), np.full(calls, chunk * k))
    ok = dropped == 0 and len(log) == calls * chunk * k and np.array_equal(hist, ref_hist.cpu().numpy()) and tokens_ok
    check("volume", ok, f"{len(log):,} records ({len(log) * 12 / 1e9:.2f} GB), histogram == torch, "
                        f"drain {t_drain:.2f}s")
    results["volume"] = dict(decisions=int(len(log)), dropped=int(dropped), drain_s=round(t_drain, 3),
                             log_loop_s=round(t_log, 3))
    del log


def test_overflow():
    print("3) overflow: 3M decisions into a 1M-record buffer")
    e, k, n, cap = 64, 8, 125_000, 1_000_000
    moescope.configure(capacity=cap, mode="softmax_topk", device="cuda")
    for c in range(3):
        moescope.route_and_log(torch.randn(n, e, device=DEV), k=k, layer_id=c, token_offset=c * n)
    log, dropped = moescope.drain_with_stats()
    valid = bool((log["expert_id"] < e).all() and (log["gate"] > 0).all() and (log["gate"] <= 1).all()
                 and (log["token_id"] < 3 * n).all())
    check("overflow: exact drop count", len(log) == cap and dropped == 3 * n * k - cap,
          f"kept {len(log):,}, dropped {dropped:,} (expected {3 * n * k - cap:,}), all kept records valid={valid}")
    check("overflow: kept records valid", valid)
    moescope.route_and_log(torch.randn(1000, e, device=DEV), k=k, layer_id=0)
    log2, dropped2 = moescope.drain_with_stats()
    check("overflow: buffer reusable after drain", len(log2) == 1000 * k and dropped2 == 0)
    for _ in range(200):  # many tiny log/drain cycles
        moescope.route_and_log(torch.randn(7, e, device=DEV), k=k, layer_id=0)
        l, d = moescope.drain_with_stats()
        if len(l) != 7 * k or d:
            break
    check("overflow: 200 log/drain cycles", len(l) == 7 * k and d == 0)


def test_adversarial():
    print("4) adversarial inputs")
    e, k, n = 64, 8, 4096
    log, _ = log_once(torch.zeros(n, e, device=DEV), k, "softmax_topk")
    ids, gates = per_token(log, n, k)
    check("all-equal logits -> lowest indices, gate 1/E", (ids == np.arange(k)).all() and np.allclose(gates, 1 / e),
          "(deterministic tie-break)")

    x = torch.randn(n, e, device=DEV)
    x[:, ::2] = float("nan")  # half the experts are NaN
    log, _ = log_once(x, k, "softmax_topk")
    ids, gates = per_token(log, n, k)
    check("NaN logits never selected", (ids % 2 == 1).all() and np.isfinite(gates).all())

    x = torch.randn(n, e, device=DEV) * 1e30
    log, _ = log_once(x, k, "topk_softmax")
    ids, gates = per_token(log, n, k)
    check("huge magnitudes (1e30): no overflow", agree(x, ids, k, "topk_softmax")[1] == 1 and np.isfinite(gates).all())

    for e2, k2 in ((32, 32), (128, 32), (2, 1), (1, 1), (65536, 8)):
        n2 = 2048 if e2 < 65536 else 64
        x = torch.randn(n2, e2, device=DEV)
        log, _ = log_once(x, k2, "softmax_topk")
        ids, gates = per_token(log, n2, k2)
        _, rgates = reference(x, k2, "softmax_topk")
        check(f"E={e2}, k={k2}", agree(x, ids, k2, "softmax_topk")[1] == 1
              and np.abs(np.sort(gates, 1) - np.sort(rgates.cpu().numpy(), 1)).max() < 1e-5)

    x = torch.randn(n, e, device=DEV)
    for dt in (torch.float16, torch.bfloat16):
        xd = x.to(dt)
        log, _ = log_once(xd, k, "softmax_topk")
        ids, _ = per_token(log, n, k)
        exact, by_value = agree(xd, ids, k, "softmax_topk")
        check(f"{dt} input", by_value == 1, f"{100 * exact:.2f}% identical ids, rest are exact {dt} ties")

    xt = torch.randn(e, n, device=DEV).t()  # non-contiguous view
    log, _ = log_once(xt, k, "softmax_topk")
    ids, _ = per_token(log, n, k)
    check("non-contiguous input", agree(xt, ids, k, "softmax_topk")[1] == 1)

    x3 = torch.randn(4, 512, e, device=DEV)  # [batch, seq, E]
    log, _ = log_once(x3, k, "softmax_topk")
    ids, _ = per_token(log, 4 * 512, k)
    check("3-D [batch, seq, E] input", agree(x3, ids, k, "softmax_topk")[1] == 1)

    side = torch.cuda.Stream()
    moescope.configure(capacity=n * k, mode="softmax_topk", device="cuda")
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)  # make the side stream slow
        xs = torch.randn(n, e, device=DEV) + 0
        moescope.route_and_log(xs, k=k, layer_id=0)
    torch.cuda.synchronize()
    log, _ = moescope.drain_with_stats()
    ids, _ = per_token(log, n, k)
    check("logits from a side stream", agree(xs, ids, k, "softmax_topk")[1] == 1)


def bench(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def test_throughput():
    print("5) throughput (one call, median of 50)")
    for e, k in ((8, 2), (32, 8), (64, 8), (128, 8)):
        for n in (4096, 65536, 1 << 20):
            x = torch.randn(n, e, device=DEV)
            moescope.configure(capacity=60 * n * k, mode="softmax_topk", device="cuda")
            ms = bench(lambda: moescope.route_and_log(x, k=k, layer_id=0))
            moescope.drain_with_stats()
            ms_torch = bench(lambda: torch.softmax(x, -1).topk(k, -1))
            row = dict(experts=e, k=k, tokens=n, moescope_ms=round(ms, 4), torch_topk_ms=round(ms_torch, 4),
                       decisions_per_s=round(n * k / ms * 1e3))
            results["throughput"].append(row)
            print(f"     E={e:<4} k={k} tokens={n:>9,}: moescope {ms:7.3f} ms "
                  f"({row['decisions_per_s'] / 1e9:.2f} G decisions/s) | torch softmax+topk alone {ms_torch:7.3f} ms")

    # DeepSeek-V3 routing (256 experts, sigmoid + bias, 8 groups, top 4 groups,
    # k = 8, normalized, x2.5) vs the same computation in torch, as written in
    # transformers' DeepseekV3TopkRouter.
    e, k, groups, top = 256, 8, 8, 4
    bias = torch.randn(e, device=DEV) * 0.1

    def torch_deepseek(x):
        scores = x.sigmoid()
        choice = scores + bias
        g = choice.view(-1, groups, e // groups).topk(2, -1)[0].sum(-1)
        keep = torch.zeros_like(g).scatter_(1, g.topk(top, -1, sorted=False)[1], 1)
        mask = keep.unsqueeze(-1).expand(-1, groups, e // groups).reshape(-1, e)
        idx = choice.masked_fill(~mask.bool(), float("-inf")).topk(k, -1, sorted=False)[1]
        w = scores.gather(1, idx)
        return idx, w / (w.sum(-1, keepdim=True) + 1e-20) * 2.5

    for n in (65536, 1 << 20):
        x = torch.randn(n, e, device=DEV)
        moescope.configure(capacity=60 * n * k, mode="sigmoid", device="cuda", normalize=True, scale=2.5,
                           n_group=groups, topk_group=top)
        moescope.set_bias(0, bias)
        ms = bench(lambda: moescope.route_and_log(x, k=k, layer_id=0))
        moescope.drain_with_stats()
        ms_torch = bench(lambda: torch_deepseek(x))
        row = dict(experts=e, k=k, tokens=n, mode="deepseek_v3", moescope_ms=round(ms, 4),
                   torch_topk_ms=round(ms_torch, 4), decisions_per_s=round(n * k / ms * 1e3))
        results["throughput"].append(row)
        print(f"     DeepSeek-V3 routing, tokens={n:>9,}: moescope {ms:7.3f} ms "
              f"({row['decisions_per_s'] / 1e9:.2f} G decisions/s) | torch (same routing) {ms_torch:7.3f} ms")

    # Half-precision logits are read in place (no float32 copy).
    for dt in (torch.float16, torch.bfloat16):
        x = torch.randn(1 << 20, 64, device=DEV).to(dt)
        moescope.configure(capacity=60 * (1 << 20) * 8, mode="softmax_topk", device="cuda")
        ms = bench(lambda: moescope.route_and_log(x, k=8, layer_id=0))
        moescope.drain_with_stats()
        print(f"     E=64   k=8 tokens=1,048,576 {str(dt)[6:]:>8} logits: moescope {ms:7.3f} ms")
        results["throughput"].append(dict(experts=64, k=8, tokens=1 << 20, dtype=str(dt)[6:], moescope_ms=round(ms, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=100_000_000, help="decisions for the volume test")
    ap.add_argument("--out", default="bench/results/stress.json")
    args = ap.parse_args()
    assert moescope.cuda_available(), "needs the CUDA build and a GPU"
    results["gpu"] = torch.cuda.get_device_name()
    print(f"moescope stress test on {results['gpu']}")
    test_correctness()
    test_volume(args.volume)
    test_overflow()
    test_adversarial()
    test_throughput()
    results["passed"] = not failures
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    if failures:
        raise SystemExit(f"FAILED: {failures}")
    print("ALL STRESS TESTS PASSED")


if __name__ == "__main__":
    main()
