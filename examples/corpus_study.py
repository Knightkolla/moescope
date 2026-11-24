"""Routing study on real corpora: logs every routing decision of an MoE model
with moescope over ~100k tokens per domain and answers four questions with
bootstrap confidence intervals (resampling 512-token chunks):

  1. Do code and maths share experts more than other domain pairs?
  2. How unbalanced is expert load per layer?
  3. Does routing depend on a token's position in its sequence?
  4. Does routing drift as the model generates a long continuation?

    python examples/corpus_study.py --model allenai/OLMoE-1B-7B-0924
    python examples/corpus_study.py --model LiquidAI/LFM2-8B-A1B --device-map auto
    python examples/corpus_study.py --toy --tokens 4000          # offline smoke test

Data (public Hugging Face datasets, no login): code = MBPP + HumanEval
solutions, maths = GSM8K, english = WikiText-103, french = French Wikipedia.
Writes <out>/study.json and <out>/study.png.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import moescope

SEQ = 512


# ---------------------------------------------------------------- data ----
def domain_texts(name, limit_chars):
    """Yields documents of one domain until ~limit_chars characters."""
    from datasets import load_dataset

    def take(it, field):
        total = 0
        for row in it:
            text = row[field] if isinstance(field, str) else field(row)
            if text and text.strip():
                total += len(text)
                yield text
                if total >= limit_chars:
                    return

    if name == "code":
        yield from take(load_dataset("openai/openai_humaneval", split="test"), lambda r: r["prompt"] + r["canonical_solution"])
        yield from take(load_dataset("google-research-datasets/mbpp", "full", split="train+test+validation"), "code")
    elif name == "maths":
        yield from take(load_dataset("openai/gsm8k", "main", split="train"), lambda r: r["question"] + "\n" + r["answer"])
    elif name == "english":
        yield from take(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True), "text")
    elif name == "french":
        yield from take(load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True), "text")


def chunks_for(tok, name, n_tokens, vocab):
    """[n_chunks, SEQ] token ids: documents concatenated, cut into SEQ chunks."""
    ids = []
    for text in domain_texts(name, limit_chars=n_tokens * 6):
        ids.extend(tok(text, add_special_tokens=False).input_ids + ([tok.eos_token_id] if tok.eos_token_id is not None else []))
        if len(ids) >= n_tokens:
            break
    n = min(len(ids), n_tokens) // SEQ
    if n == 0:
        raise SystemExit(f"not enough {name} text")
    return torch.tensor(ids[: n * SEQ]).view(n, SEQ)


def toy_chunks(name, n_tokens, vocab):
    g = torch.Generator().manual_seed(hash(name) % 2**31)
    lo = {"code": 3, "maths": 60, "english": 120, "french": 180}[name]
    n = n_tokens // SEQ
    return torch.randint(lo, lo + 60, (n, SEQ), generator=g) % vocab


# ------------------------------------------------------------ analysis ----
def counts(log, num_layers, num_experts):
    """[layers, experts] number of decisions."""
    flat = log["layer_id"].astype(np.int64) * num_experts + log["expert_id"]
    return np.bincount(flat, minlength=num_layers * num_experts).reshape(num_layers, num_experts).astype(np.float64)


def normalize(c):
    return c / np.maximum(c.sum(-1, keepdims=True), 1)


def usage(log, num_layers, num_experts):
    """[layers, experts] share of decisions."""
    return normalize(counts(log, num_layers, num_experts))


def js_bits(p, q):
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def top_overlap(a, b, n=8):
    """Jaccard overlap of the n most-used experts."""
    ta, tb = set(np.argsort(-a)[:n]), set(np.argsort(-b)[:n])
    return len(ta & tb) / len(ta | tb)


def bootstrap(stat, groups, reps=200, seed=0):
    """stat(list_of_index_arrays) -> float; resamples chunks within each group."""
    rng = np.random.default_rng(seed)
    vals = [stat([rng.choice(g, len(g)) for g in groups]) for _ in range(reps)]
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--tokens", type=int, default=100_000, help="tokens per domain")
    ap.add_argument("--gen-tokens", type=int, default=512, help="tokens generated per prompt for the drift test")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--batch", type=int, default=8, help="chunks per forward pass")
    ap.add_argument("--out", default="study")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.toy:
        import transformers as T

        torch.manual_seed(0)
        cfg = T.OlmoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=4, num_experts=16, num_experts_per_tok=4,
                            max_position_embeddings=2048, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model, tok, name = T.OlmoeForCausalLM(cfg).to(device).eval(), None, "toy-olmoe"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device_map or device).eval()
        name = args.model.split("/")[-1]
    vocab = model.config.vocab_size
    in_device = model.get_input_embeddings().weight.device

    domains = ["code", "maths", "english", "french"]
    data = {}
    for d in domains:
        t0 = time.time()
        data[d] = toy_chunks(d, args.tokens, vocab) if args.toy else chunks_for(tok, d, args.tokens, vocab)
        print(f"data {d}: {data[d].numel():,} tokens in {time.time() - t0:.0f}s", flush=True)
    spec = moescope.routing_spec(model.config)

    # --- prefill over the corpus: one log, chunk -> domain bookkeeping -----
    chunk_domain = []
    n_layers = len(moescope.hf.find_routers(model, spec.num_experts))
    capacity = sum(len(c) for c in data.values()) * SEQ * n_layers * spec.k
    t0 = time.time()
    with moescope.attach(model, capacity=capacity) as session:
        with torch.no_grad():
            for di, d in enumerate(domains):
                for b in range(0, len(data[d]), args.batch):
                    batch = data[d][b : b + args.batch].to(in_device)
                    model(input_ids=batch)
                    chunk_domain += [di] * len(batch)
                print(f"prefill {d} done ({time.time() - t0:.0f}s)", flush=True)
        log, dropped = session.drain_with_stats()
    t_prefill = time.time() - t0
    layer_ids = sorted(session.routers)
    L, E = len(layer_ids), spec.num_experts
    log = log.copy()
    log["layer_id"] = np.searchsorted(layer_ids, log["layer_id"])
    chunk = (log["token_id"] // SEQ).astype(np.int64)
    pos = log["token_id"] % SEQ
    chunk_domain = np.array(chunk_domain)
    n_chunks = len(chunk_domain)
    # Per-chunk decision counts [chunk, layer, expert]: every statistic below
    # (and each bootstrap resample) is a sum over chunks of this tensor.
    flat = (chunk * L + log["layer_id"]) * E + log["expert_id"]
    per_chunk = np.bincount(flat, minlength=n_chunks * L * E).reshape(n_chunks, L, E).astype(np.float64)
    print(f"{name}: {len(log):,} decisions over {n_chunks * SEQ:,} tokens, dropped={dropped}, {t_prefill:.0f}s")

    def domain_usage(chunk_ids):
        return normalize(per_chunk[chunk_ids].sum(0))

    groups = [np.flatnonzero(chunk_domain == i) for i in range(len(domains))]

    # 1) domain pairs: top-8 expert overlap per layer, mean over layers + CI
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    overlap = {}
    for a, b in pairs:
        def stat(gs, a=a, b=b):
            ua, ub = domain_usage(gs[0]), domain_usage(gs[1])
            return float(np.mean([top_overlap(ua[l], ub[l]) for l in range(L)]))

        overlap[f"{domains[a]}-{domains[b]}"] = bootstrap(stat, [groups[a], groups[b]], reps=100)
    full = {d: domain_usage(groups[i]) for i, d in enumerate(domains)}
    per_layer_overlap = {f"{domains[a]}-{domains[b]}": [top_overlap(full[domains[a]][l], full[domains[b]][l]) for l in range(L)]
                         for a, b in pairs}

    # 2) load imbalance (busiest expert / fair share) per layer, all domains
    all_usage = usage(log, L, E)
    imbalance = (all_usage.max(1) * E).tolist()
    idle = (all_usage < 0.1 / E).sum(1).tolist()
    imb_ci = bootstrap(lambda gs: float(np.mean(domain_usage(np.concatenate(gs)).max(1) * E)), groups, reps=100)

    # 3) position: JS divergence between early (<64) and late (>=256) positions, per layer
    early, late = usage(log[pos < 64], L, E), usage(log[pos >= 256], L, E)
    pos_js = [js_bits(early[l], late[l]) for l in range(L)]
    half = len(chunk_domain) // 2  # noise floor: same positions, two halves of the corpus
    split = chunk < half
    floor_js = [js_bits(usage(log[split & (pos >= 256)], L, E)[l], usage(log[~split & (pos >= 256)], L, E)[l]) for l in range(L)]

    # 4) generation drift: routing of generated tokens, first vs last quarter
    drift = None
    if args.gen_tokens > 0 and not args.toy:
        print(f"generating {args.gen_tokens} tokens x 8 prompts", flush=True)
        prompts = torch.cat([data[d][:2, :128] for d in domains]).to(in_device)  # 8 prompts, 2 per domain
        gen_cap = (prompts.numel() + args.gen_tokens * len(prompts)) * n_layers * spec.k
        with moescope.attach(model, capacity=gen_cap) as gen:
            with torch.no_grad():
                model.generate(prompts, max_new_tokens=args.gen_tokens, min_new_tokens=args.gen_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            glog = gen.drain()
        glog = glog.copy()
        glog["layer_id"] = np.searchsorted(layer_ids, glog["layer_id"])
        gen_ids = glog["token_id"] - prompts.numel()  # decode tokens only (prefill first)
        step = gen_ids // len(prompts)
        g = glog[gen_ids >= 0]
        step = step[gen_ids >= 0]
        q = args.gen_tokens // 4
        first, last = usage(g[step < q], L, E), usage(g[step >= 3 * q], L, E)
        drift = {"js_first_vs_last_quarter_per_layer": [js_bits(first[l], last[l]) for l in range(L)],
                 "prefill_vs_generated_js_per_layer": [js_bits(all_usage[l], usage(g, L, E)[l]) for l in range(L)]}

    result = {
        "model": name, "tokens_per_domain": {d: int(len(data[d]) * SEQ) for d in domains},
        "decisions": int(len(log)), "dropped": int(dropped), "moe_layer_ids": layer_ids, "experts": E, "k": spec.k,
        "top8_overlap_mean_ci": overlap, "top8_overlap_per_layer": per_layer_overlap,
        "imbalance_per_layer": imbalance, "imbalance_mean_ci": imb_ci, "idle_experts_per_layer": idle,
        "position_js_early_vs_late_per_layer": pos_js, "position_js_noise_floor_per_layer": floor_js,
        "generation_drift": drift,
    }
    with open(os.path.join(args.out, "study.json"), "w") as f:
        json.dump(result, f, indent=1)
    plot(result, domains, os.path.join(args.out, "study.png"))
    print(json.dumps({k: v for k, v in result.items() if "per_layer" not in k}, indent=1))


def plot(r, domains, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    layers = np.arange(len(r["imbalance_per_layer"]))
    for name, vals in r["top8_overlap_per_layer"].items():
        ax[0].plot(layers, vals, marker="o", ms=3, lw=2.2 if name == "code-maths" else 1, label=name)
    ax[0].set(title="Top-8 expert overlap between domains", xlabel="MoE layer", ylabel="Jaccard overlap")
    ax[0].legend(fontsize=8)
    ax[1].bar(layers, r["imbalance_per_layer"], color="#c44e52")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set(title="Busiest expert vs fair share", xlabel="MoE layer", ylabel="x fair share")
    ax[2].plot(layers, r["position_js_early_vs_late_per_layer"], marker="o", ms=3, label="positions <64 vs >=256")
    ax[2].plot(layers, r["position_js_noise_floor_per_layer"], ls="--", label="noise floor (corpus halves)")
    if r["generation_drift"]:
        ax[2].plot(layers, r["generation_drift"]["js_first_vs_last_quarter_per_layer"], marker="s", ms=3,
                   label="generation: first vs last quarter")
    ax[2].set(title="Routing shift (JS divergence, bits)", xlabel="MoE layer")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"{r['model']}: {r['decisions']:,} routing decisions over "
                 f"{sum(r['tokens_per_domain'].values()):,} tokens, logged with moescope", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
