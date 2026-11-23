"""Routing report for a real MoE model: log every router decision with
moescope, then summarise expert load, dead experts, router confidence and
per-domain expert specialisation, and save a figure.

    python examples/analyze_routing.py                       # Granite-3.1-1B-A400M (32 experts, top-8)
    python examples/analyze_routing.py --model allenai/OLMoE-1B-7B-0924
    python examples/analyze_routing.py --toy granitemoe      # tiny random model, offline smoke test

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
import torch.nn.functional as F

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

# Where each architecture keeps its router (first match wins).
ROUTER_PATHS = ["mlp.gate", "block_sparse_moe.router", "block_sparse_moe.gate", "mlp.router"]


def build_model(args):
    if args.toy:
        return build_toy(args.toy, args.device), None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
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
    cfg = T.MixtralConfig(num_local_experts=8, num_experts_per_tok=2, **common)
    return T.MixtralForCausalLM(cfg).to(device).eval()


def routing_spec(config):
    """(num_experts, k, moescope mode) for a model config."""
    e = getattr(config, "num_experts", None) or getattr(config, "num_local_experts")
    k = config.num_experts_per_tok
    if config.model_type in ("mixtral", "granitemoe"):
        mode = "topk_softmax"  # top-k on logits, softmax over the k selected
    else:
        mode = "topk_softmax" if getattr(config, "norm_topk_prob", False) else "softmax_topk"
    return e, k, mode


def find_routers(model):
    routers = []
    for layer in model.model.layers:
        for path in ROUTER_PATHS:
            mod = layer
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    break
            if mod is not None:
                routers.append(mod)
                break
        else:
            raise SystemExit(f"no MoE router found in {type(layer).__name__}; add its path to ROUTER_PATHS")
    return routers


def router_weight(router):
    """[num_experts, hidden] weight of a router (module.weight or a wrapped Linear)."""
    if isinstance(getattr(router, "weight", None), torch.Tensor):
        return router.weight
    return router.layer.weight  # transformers 4.x GraniteMoeTopKGating


def tokenize(tok, text, vocab_size, device, seq_cap=512):
    if tok is None:  # toy model: deterministic pseudo-tokens from the bytes
        ids = torch.tensor([b % (vocab_size - 3) + 3 for b in text.encode()][:seq_cap])
    else:
        ids = tok(text, return_tensors="pt").input_ids[0][:seq_cap]
    return ids.unsqueeze(0).to(device)


class RoutingLogger:
    """Hooks every router: logs decisions with moescope and (optionally)
    keeps the model's own top-k choice for a parity check."""

    def __init__(self, routers, k, check_parity):
        self.routers, self.k, self.check = routers, k, check_parity
        self.offset, self.handles, self.reference = 0, [], {}

    def attach(self):
        for layer_id, r in enumerate(self.routers):
            self.handles.append(r.register_forward_pre_hook(self._pre(layer_id, r)))
            if self.check:
                self.handles.append(r.register_forward_hook(self._post(layer_id)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre(self, layer_id, router):
        w = router_weight(router)

        def hook(_mod, inputs):
            h = inputs[0]
            # Same op the router itself runs, so the logits are bit-identical.
            logits = F.linear(h.reshape(-1, h.shape[-1]), w).float()
            moescope.route_and_log(logits, k=self.k, layer_id=layer_id, token_offset=self.offset)
        return hook

    def _post(self, layer_id):
        def hook(_mod, _inp, out):
            # transformers >= 5 routers return (logits, weights, indices).
            if isinstance(out, tuple) and len(out) >= 3 and out[2].dtype in (torch.int64, torch.int32):
                idx = out[2].reshape(-1, self.k)
                self.reference[(layer_id, self.offset)] = idx.cpu()
        return hook


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


def parity(log, reference, k):
    """Fraction of (layer, token) decisions whose expert set equals the model's."""
    if not reference:
        return None
    by_key = {}
    srt = log[np.lexsort((log["expert_id"], log["token_id"], log["layer_id"]))]
    ids = srt["expert_id"].reshape(-1, k)
    keys = np.stack([srt["layer_id"][::k], srt["token_id"][::k]], 1)
    for row, (l, t) in zip(ids, keys):
        by_key[(int(l), int(t))] = row
    match = total = 0
    for (layer, offset), idx in reference.items():
        ref = np.sort(idx.numpy(), axis=1)
        for i, r in enumerate(ref):
            total += 1
            match += np.array_equal(by_key[(layer, offset + i)], r)
    return match / total, total


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
    ap.add_argument("--toy", choices=["granitemoe", "olmoe", "mixtral"], default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeat", type=int, default=8, help="passes over the domain texts (more tokens)")
    ap.add_argument("--out", default="routing_report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = build_model(args)
    num_experts, k, mode = routing_spec(model.config)
    routers = find_routers(model)
    num_layers = len(routers)
    name = f"toy-{args.toy}" if args.toy else args.model.split("/")[-1]
    backend = "cuda" if args.device == "cuda" and moescope.cuda_available() else "cpu"
    batches = [tokenize(tok, text, model.config.vocab_size, args.device) for text in DOMAINS.values()]
    tokens_per_pass = sum(b.shape[1] for b in batches)
    capacity = tokens_per_pass * args.repeat * num_layers * k
    rec = moescope.configure(capacity=capacity, mode=mode, device=backend)
    print(f"{name}: {num_layers} MoE layers, {num_experts} experts, top-{k}, mode={mode}, moescope backend={rec.backend}")

    # 1) Logging cost: same forward passes with and without the moescope hooks.
    logger = RoutingLogger(routers, k, check_parity=False)
    time_forward(model, batches, 2, args.device)  # warm-up
    t_plain = time_forward(model, batches, 5, args.device)
    logger.attach()
    t_logged = time_forward(model, batches, 5, args.device)
    logger.detach()
    moescope.drain_with_stats()  # discard timing-run records (may overflow; irrelevant)

    # 2) The real logging run (+ parity check against the model's own router).
    logger = RoutingLogger(routers, k, check_parity=True)
    logger.attach()
    domain_of_token = []
    with torch.no_grad():
        for _ in range(args.repeat):
            for d, ids in enumerate(batches):
                model(input_ids=ids)
                logger.offset += ids.shape[1]
                domain_of_token += [d] * ids.shape[1]
    logger.detach()
    log, dropped = moescope.drain_with_stats()
    domain_of_token = np.array(domain_of_token)
    assert dropped == 0 and len(log) == logger.offset * num_layers * k, (dropped, len(log))

    stats = analyse(log, domain_of_token, num_layers, num_experts, k)
    par = parity(log, logger.reference, k)
    title = f"{name}: routing of {logger.offset:,} tokens x {num_layers} layers ({len(log):,} decisions), logged with moescope"
    plot(stats, title, os.path.join(args.out, "routing_report.png"))
    np.savez_compressed(os.path.join(args.out, "routing_log.npz"), log=log, domain_of_token=domain_of_token,
                        domains=np.array(stats["domains"]))

    summary = {
        "model": name, "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "moescope_backend": rec.backend, "layers": num_layers, "experts": num_experts, "k": k, "mode": mode,
        "tokens": int(logger.offset), "decisions": int(len(log)), "dropped": int(dropped),
        "forward_ms_plain": round(t_plain, 2), "forward_ms_logged": round(t_logged, 2),
        "logging_overhead_pct": round(100 * (t_logged - t_plain) / t_plain, 1),
        "parity_with_model_router": None if par is None else {"match_rate": par[0], "decisions_checked": par[1]},
        "imbalance_per_layer": np.round(stats["imbalance"], 2).tolist(),
        "idle_experts_per_layer": stats["dead"].tolist(),
        "mean_top1_gate_per_layer": np.round(stats["top1"], 3).tolist(),
        "specialisation_bits_per_layer": np.round(stats["specialisation"], 3).tolist(),
        "most_specialised_layer": int(np.argmax(stats["specialisation"])),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nlogged {len(log):,} routing decisions ({logger.offset:,} tokens x {num_layers} layers x top-{k}), dropped={dropped}")
    print(f"forward pass: {t_plain:.1f} ms plain vs {t_logged:.1f} ms with moescope logging "
          f"({summary['logging_overhead_pct']:+.1f}%)")
    if par is not None:
        print(f"parity: {100 * par[0]:.2f}% of {par[1]:,} (layer, token) expert sets match the model's own router")
    print(f"load imbalance (busiest expert / fair share): min {stats['imbalance'].min():.2f}, "
          f"max {stats['imbalance'].max():.2f} (layer {int(stats['imbalance'].argmax())})")
    print(f"idle experts (<10% of fair share): {int(stats['dead'].sum())} across all layers")
    print(f"most domain-specialised layer: {summary['most_specialised_layer']} "
          f"({stats['specialisation'].max():.3f} bits JS); least: {int(stats['specialisation'].argmin())}")
    print(f"\nwrote {args.out}/routing_report.png, routing_log.npz, summary.json")


if __name__ == "__main__":
    main()
