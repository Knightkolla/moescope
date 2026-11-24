"""Routing report for a real MoE model: log every router decision with
moescope, then summarise expert load, dead experts, router confidence and
per-domain expert specialisation, and save a figure.

    python examples/analyze_routing.py                       # Granite-3.1-1B-A400M (32 experts, top-8)
    python examples/analyze_routing.py --model allenai/OLMoE-1B-7B-0924
    python examples/analyze_routing.py --model LiquidAI/LFM2-8B-A1B --device-map auto   # sigmoid + bias, 2 GPUs
    python examples/analyze_routing.py --toy deepseek_v3     # tiny random model, offline smoke test

Outputs (in --out, default ./routing_report/):
    routing_report.png   heatmaps + per-layer stats
    routing_log.npz      the raw log (token_id, layer_id, expert_id, gate) + token->domain map
    summary.json         the numbers printed at the end
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import moescope

# Short original passages, one per domain. Routing differences between them
# are what the specialisation panel shows.
DOMAINS = {
    "code": '''
def merge_intervals(intervals):
    """Merge overlapping [start, end] intervals and return them sorted."""
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store = {}
    def get(self, key):
        if key not in self.store:
            return -1
        value = self.store.pop(key)
        self.store[key] = value
        return value
    def put(self, key, value):
        self.store.pop(key, None)
        if len(self.store) >= self.capacity:
            self.store.pop(next(iter(self.store)))
        self.store[key] = value

for i in range(10):
    print(i, merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]))
''',
    "math": r'''
Let f(x) = x^3 - 6x^2 + 11x - 6. Since f(1) = 0, the polynomial factors as
(x - 1)(x^2 - 5x + 6) = (x - 1)(x - 2)(x - 3), so the roots are 1, 2 and 3.
The derivative is f'(x) = 3x^2 - 12x + 11, which vanishes at x = 2 ± 1/sqrt(3).
For the integral, \int_0^1 x e^{x} dx = [x e^x]_0^1 - \int_0^1 e^x dx = e - (e - 1) = 1.
If a_n = 1/n^2 then the series \sum_{n=1}^{\infty} a_n converges to \pi^2 / 6.
Consider the matrix A = [[2, 1], [1, 2]]. Its characteristic polynomial is
\lambda^2 - 4\lambda + 3, so the eigenvalues are \lambda = 1 and \lambda = 3,
with eigenvectors (1, -1) and (1, 1). Therefore A^n = P D^n P^{-1} where D = diag(1, 3).
By the Cauchy-Schwarz inequality, |<u, v>| <= ||u|| ||v|| for all vectors u and v.
''',
    "prose": '''
The ferry left the harbour just after dawn, its engines grumbling against the
cold. Mara stood at the rail with her coat buttoned to the chin, watching the
town shrink into a line of grey roofs and one stubborn lighthouse. She had
lived there for eleven years and had never once taken this crossing, though
she had watched the boat come and go nearly every morning from her kitchen
window. Now, with a single suitcase and a letter she had read so many times
the folds had gone soft, she wondered why she had waited so long. The gulls
followed for a while, then gave up. Somewhere below, a child was laughing.
By the time the island appeared, low and green under a sky that could not
decide what it wanted to be, she had made up her mind about almost everything.
''',
    "french": '''
Le petit marché du samedi s'installe toujours sur la place de l'église, bien
avant que la ville ne se réveille. Les marchands déplient leurs tables, disposent
les tomates, les fromages et les bouquets de lavande, puis échangent les
nouvelles de la semaine en buvant un café trop chaud. Vers neuf heures, les
premiers clients arrivent avec leurs paniers. On discute du temps qu'il fera,
du prix des cerises et du concert prévu le mois prochain. Les enfants courent
entre les étals pendant que les chiens attendent patiemment près de la
boulangerie. À midi, tout est rangé, la place redevient silencieuse, et il ne
reste qu'une odeur de pain chaud et quelques pétales oubliés sur les pavés.
''',
}

def build_model(args):
    if args.toy:
        return build_toy(args.toy, args.device), None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    extra = {}
    if args.load_4bit:
        # Expert/attention matmuls in 4-bit (fp16 compute); everything that
        # is not quantized, including the router weights, stays fp32, so
        # routing is computed exactly as in full precision.
        from transformers import BitsAndBytesConfig

        extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        dtype = torch.float32
    # device_map loads weights straight onto the GPU: OLMoE-1B-7B is ~14 GB in
    # fp16, more than a free Colab VM's RAM, but it fits a 16 GB T4.
    device_map = args.device_map or args.device
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device_map, **extra).eval()
    return model, tok


def build_toy(kind, device):
    """Tiny random-weight model with the real HF architecture (offline test)."""
    import transformers as T

    torch.manual_seed(0)
    common = dict(vocab_size=512, hidden_size=128, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                  bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if kind == "granitemoe":
        cfg = T.GraniteMoeConfig(num_local_experts=16, num_experts_per_tok=4, **common)
        return T.GraniteMoeForCausalLM(cfg).to(device).eval()
    if kind == "olmoe":
        cfg = T.OlmoeConfig(num_experts=16, num_experts_per_tok=4, **common)
        return T.OlmoeForCausalLM(cfg).to(device).eval()
    if kind == "deepseek_v3":
        cfg = T.DeepseekV3Config(n_routed_experts=32, num_experts_per_tok=4, n_group=4, topk_group=2,
                                 norm_topk_prob=True, routed_scaling_factor=2.5, moe_intermediate_size=64,
                                 first_k_dense_replace=1, q_lora_rank=None, kv_lora_rank=32, qk_nope_head_dim=16,
                                 qk_rope_head_dim=16, v_head_dim=32, **common)
        return T.DeepseekV3ForCausalLM(cfg).to(device).eval()
    cfg = T.MixtralConfig(num_local_experts=8, num_experts_per_tok=2, **common)
    return T.MixtralForCausalLM(cfg).to(device).eval()


def tokenize(tok, text, vocab_size, device, seq_cap=512):
    if tok is None:  # toy model: deterministic pseudo-tokens from the bytes
        ids = torch.tensor([b % (vocab_size - 3) + 3 for b in text.encode()][:seq_cap])
    else:
        ids = tok(text, return_tensors="pt").input_ids[0][:seq_cap]
    return ids.unsqueeze(0).to(device)


def analyse(log, domain_of_token, num_layers, num_experts, k):
    load = np.zeros((num_layers, num_experts), np.int64)
    np.add.at(load, (log["layer_id"].astype(int), log["expert_id"].astype(int)), 1)
    share = load / load.sum(axis=1, keepdims=True)
    uniform = 1.0 / num_experts
    imbalance = share.max(axis=1) / uniform              # 1.0 = perfectly balanced
    dead = (share < 0.1 * uniform).sum(axis=1)           # experts getting <10% of a fair share
    # Router confidence: mean of each token's largest gate, per layer.
    order = np.lexsort((-log["gate"], log["token_id"], log["layer_id"]))
    top1 = log[order][::k]["gate"]
    top1_by_layer = np.array([top1[log[order][::k]["layer_id"] == l].mean() for l in range(num_layers)])

    domains = list(DOMAINS)
    dom = domain_of_token[log["token_id"]]
    usage = np.zeros((len(domains), num_layers, num_experts))
    for d in range(len(domains)):
        sel = dom == d
        np.add.at(usage[d], (log["layer_id"][sel].astype(int), log["expert_id"][sel].astype(int)), 1)
    usage /= usage.sum(axis=2, keepdims=True)
    # Specialisation per layer: mean Jensen-Shannon divergence (bits) between domain usage vectors.
    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    def js(p, q):
        m = (p + q) / 2
        return (kl(p, m) + kl(q, m)) / 2
    spec = np.array([np.mean([js(usage[a, l], usage[b, l]) for a in range(len(domains)) for b in range(a + 1, len(domains))])
                     for l in range(num_layers)])
    return dict(load=load, share=share, imbalance=imbalance, dead=dead, top1=top1_by_layer,
                usage=usage, specialisation=spec, domains=domains)


def plot(stats, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers, num_experts = stats["share"].shape
    best = int(np.argmax(stats["specialisation"]))
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, :2])
    im = ax.imshow(stats["share"] * num_experts, aspect="auto", cmap="magma", vmin=0, vmax=max(3, np.percentile(stats["share"] * num_experts, 99)))
    ax.set(xlabel="expert", ylabel="layer", title="Expert load (1.0 = fair share)")
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 12)))
    fig.colorbar(im, ax=ax, fraction=0.025)

    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(num_layers)
    ax.barh(y, stats["imbalance"], color="#d1495b", label="max load / fair share")
    ax.set(xlabel="busiest expert vs fair share", ylabel="layer", title="Load imbalance per layer")
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 12)))
    ax.invert_yaxis()
    for l in range(num_layers):
        if stats["dead"][l]:
            ax.text(stats["imbalance"][l], l, f"  {stats['dead'][l]} idle", va="center", fontsize=8)

    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(stats["usage"][:, best, :] * num_experts, aspect="auto", cmap="viridis")
    ax.set(yticks=range(len(stats["domains"])), yticklabels=stats["domains"], xlabel="expert",
           title=f"Who handles what: per-domain expert usage, layer {best} (most specialised)")
    fig.colorbar(im, ax=ax, fraction=0.025)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(stats["specialisation"], y, "o-", color="#00798c")
    ax.set_xlabel("domain specialisation (JS divergence, bits)", color="#00798c")
    ax.tick_params(axis="x", colors="#00798c")
    ax.set(ylabel="layer", title="Specialisation & router confidence")
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 12)))
    ax.invert_yaxis()
    ax2 = ax.twiny()
    ax2.plot(stats["top1"], y, "s--", color="#c9822b")
    ax2.set_xlabel("mean top-1 gate", color="#c9822b")
    ax2.tick_params(axis="x", colors="#c9822b")
    fig.savefig(path, dpi=150)


def time_forward(model, batches, reps, device):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(reps):
            sync()
            t0 = time.perf_counter()
            for ids in batches:
                model(input_ids=ids)
            sync()
            times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--toy", choices=["granitemoe", "olmoe", "mixtral", "deepseek_v3"], default="")
    ap.add_argument("--load-4bit", action="store_true",
                    help="bitsandbytes 4-bit nn.Linear weights (routers stay fp32; fused MoE experts are not quantized)")
    ap.add_argument("--device-map", default=None, help='e.g. "auto" to split a model over several GPUs')
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeat", type=int, default=8, help="passes over the domain texts (more tokens)")
    ap.add_argument("--out", default="routing_report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = build_model(args)
    spec = moescope.routing_spec(model.config)
    num_experts, k = spec.num_experts, spec.k
    name = f"toy-{args.toy}" if args.toy else args.model.split("/")[-1]
    backend = "cuda" if args.device == "cuda" and moescope.cuda_available() else "cpu"
    batches = [tokenize(tok, text, model.config.vocab_size, args.device) for text in DOMAINS.values()]
    tokens_per_pass = sum(b.shape[1] for b in batches)

    # 1) Logging cost: same forward passes with and without the moescope hooks.
    time_forward(model, batches, 2, args.device)  # warm-up
    t_plain = time_forward(model, batches, 5, args.device)
    with moescope.attach(model, device=backend, capacity=1 << 24) as timing:
        t_logged = time_forward(model, batches, 5, args.device)
    timing.drain_with_stats()  # discard timing-run records

    # 2) The real logging run (+ parity check against the model's own router).
    layer_ids = sorted(timing.routers)
    num_layers = len(layer_ids)
    capacity = tokens_per_pass * args.repeat * num_layers * k
    session = moescope.attach(model, device=backend, capacity=capacity, check=True)
    print(f"{name}: {num_layers} MoE layers, {num_experts} experts, top-{k}, mode={spec.mode} "
          f"(normalize={spec.normalize}, scale={spec.scale}, groups={spec.n_group}/{spec.topk_group}), "
          f"moescope backend={session.recorder.backend}")
    domain_of_token = []
    with session, torch.no_grad():
        for _ in range(args.repeat):
            for d, ids in enumerate(batches):
                model(input_ids=ids)
                domain_of_token += [d] * ids.shape[1]
    log, dropped = session.drain_with_stats()
    domain_of_token = np.array(domain_of_token)
    assert dropped == 0 and len(log) == session.offset * num_layers * k, (dropped, len(log))
    par = session.parity(log)
    # Dense layers (e.g. DeepSeek's first ones) have no router: analyse MoE
    # layers by position; layer_ids maps positions back to model layers.
    log = log.copy()
    log["layer_id"] = np.searchsorted(layer_ids, log["layer_id"])

    stats = analyse(log, domain_of_token, num_layers, num_experts, k)
    title = f"{name}: routing of {session.offset:,} tokens x {num_layers} layers ({len(log):,} decisions), logged with moescope"
    plot(stats, title, os.path.join(args.out, "routing_report.png"))
    np.savez_compressed(os.path.join(args.out, "routing_log.npz"), log=log, domain_of_token=domain_of_token,
                        domains=np.array(stats["domains"]))

    summary = {
        "model": name, "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "moescope_backend": session.recorder.backend, "layers": num_layers, "experts": num_experts, "k": k,
        "tokens": int(session.offset), "decisions": int(len(log)), "dropped": int(dropped),
        "forward_ms_plain": round(t_plain, 2), "forward_ms_logged": round(t_logged, 2),
        "logging_overhead_pct": round(100 * (t_logged - t_plain) / t_plain, 1),
        "parity_with_model_router": par, "moe_layer_ids": layer_ids,
        "gating": {"mode": spec.mode, "normalize": spec.normalize, "scale": spec.scale,
                   "n_group": spec.n_group, "topk_group": spec.topk_group},
        "imbalance_per_layer": np.round(stats["imbalance"], 2).tolist(),
        "idle_experts_per_layer": stats["dead"].tolist(),
        "mean_top1_gate_per_layer": np.round(stats["top1"], 3).tolist(),
        "specialisation_bits_per_layer": np.round(stats["specialisation"], 3).tolist(),
        "most_specialised_layer": layer_ids[int(np.argmax(stats["specialisation"]))],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nlogged {len(log):,} routing decisions ({session.offset:,} tokens x {num_layers} layers x top-{k}), dropped={dropped}")
    print(f"forward pass: {t_plain:.1f} ms plain vs {t_logged:.1f} ms with moescope logging "
          f"({summary['logging_overhead_pct']:+.1f}%)")
    if par is not None:
        print(f"parity: {100 * par['match_rate']:.2f}% of {par['checked']:,} (layer, token) expert sets match "
              f"the model's own router; max gate difference {par['max_gate_err']:.1e}")
    print(f"load imbalance (busiest expert / fair share): min {stats['imbalance'].min():.2f}, "
          f"max {stats['imbalance'].max():.2f} (layer {layer_ids[int(stats['imbalance'].argmax())]})")
    print(f"idle experts (<10% of fair share): {int(stats['dead'].sum())} across all layers")
    print(f"most domain-specialised layer: {summary['most_specialised_layer']} "
          f"({stats['specialisation'].max():.3f} bits JS); least: {layer_ids[int(stats['specialisation'].argmin())]}")
    print(f"\nwrote {args.out}/routing_report.png, routing_log.npz, summary.json")


if __name__ == "__main__":
    main()
