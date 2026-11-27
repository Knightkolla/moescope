"""How many experts does a workload actually need? Plan from a routing log,
then measure what pruning costs.

1. Log routing (moescope.attach) on planning text of two domains.
2. Per layer, keep the experts that carry 90% / 99% / 99.9% of the target
   domain's gate mass (moescope.analysis.experts_for_coverage).
3. On held-out target text, measure perplexity when the router may only
   pick kept experts (it falls back to its next-best allowed choice), against
   the full model, a random set of the same size per layer, and a set planned
   on the other domain.
4. Report how much of the held-out routing a GPU-resident budget of the
   busiest experts serves (for CPU offload).

    python examples/expert_budget.py --model allenai/OLMoE-1B-7B-0924 --target code --other french
    python examples/expert_budget.py --toy                        # offline smoke test

Writes <out>/budget.json and <out>/budget.png.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import moescope
from moescope.analysis import budget_curve, coverage_table, expert_usage, experts_for_coverage

sys.path.insert(0, os.path.dirname(__file__))
from corpus_study import SEQ, chunks_for, toy_chunks  # noqa: E402


def log_usage(model, chunks, spec, batch, in_device):
    """[layers, experts] gate mass and counts over `chunks` (full model)."""
    n_layers = len(moescope.hf.find_routers(model, spec.num_experts))
    mass = counts = None
    with moescope.attach(model, capacity=batch * SEQ * n_layers * spec.k) as s:
        with torch.no_grad():
            for b in range(0, len(chunks), batch):
                model(input_ids=chunks[b : b + batch].to(in_device), logits_to_keep=1)
                log = s.drain()
                ids, m = expert_usage(log, spec.num_experts, "gate")
                _, c = expert_usage(log, spec.num_experts, "count")
                mass = m if mass is None else mass + m
                counts = c if counts is None else counts + c
    return ids, mass, counts


class Restrict:
    """Re-routes every MoE layer so only `allowed[layer]` experts can be
    picked: hooks the router (to see its logits) and rewrites the
    (top_k_index, top_k_weights) the experts receive, following the model's
    routing rule with the disallowed experts removed."""

    def __init__(self, model, spec):
        self.spec, self.allowed, self.handles, self.logits = spec, None, [], {}
        routers = moescope.hf.find_routers(model, spec.num_experts)
        for lid, r in routers.items():
            block = moescope.hf._walk(r._moescope_layer, r._moescope_path.rsplit(".", 1)[0])
            self.handles.append(r.register_forward_hook(self._grab(lid)))
            self.handles.append(block.experts.register_forward_pre_hook(self._reroute(lid), with_kwargs=True))
            bias = moescope.hf.selection_bias(r._moescope_layer, r._moescope_path)
            if bias is not None:
                object.__setattr__(r, "_budget_bias", bias)  # not a submodule
        self.routers = routers

    def _grab(self, lid):
        spec = self.spec

        def hook(_m, _a, out):
            logits, _, _ = moescope.hf._pick(out, spec)
            self.logits[lid] = logits.reshape(-1, spec.num_experts)

        return hook

    def _reroute(self, lid):
        spec = self.spec

        def pre(_m, args, kwargs):
            if self.allowed is None:
                return None
            h, idx, w = args[0], args[1], args[2]
            x = self.logits[lid].float()
            mask = torch.full((spec.num_experts,), float("-inf"), device=x.device)
            mask[torch.as_tensor(self.allowed[lid], device=x.device)] = 0.0
            if spec.mode == "sigmoid":
                score = torch.sigmoid(x)
                bias = getattr(self.routers[lid], "_budget_bias", None)
                choice = score + (bias.float() if bias is not None else 0.0)
                sel = (choice + mask).topk(spec.k, -1).indices
                g = score.gather(1, sel)
                if spec.normalize:
                    g = g / (g.sum(-1, keepdim=True) + 1e-20)
                g = g * spec.scale
            elif spec.mode == "softmax_topk":
                p = torch.softmax(x, -1)
                sel = (x + mask).topk(spec.k, -1).indices
                g = p.gather(1, sel)
            else:  # topk_softmax
                sel = (x + mask).topk(spec.k, -1).indices
                g = torch.softmax(x.gather(1, sel), -1)
            return (h, sel.to(idx.dtype).view_as(idx), g.to(w.dtype).view_as(w)), kwargs

        return pre

    def remove(self):
        for h in self.handles:
            h.remove()


def perplexity(model, chunks, batch, in_device):
    nll, n = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(chunks), batch):
            x = chunks[b : b + batch].to(in_device)
            out = model(input_ids=x, labels=x)
            t = x.shape[0] * (x.shape[1] - 1)
            nll += float(out.loss) * t
            n += t
    return float(np.exp(nll / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--target", default="code")
    ap.add_argument("--other", default="french")
    ap.add_argument("--plan-tokens", type=int, default=80_000)
    ap.add_argument("--eval-tokens", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", default="budget")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.toy:
        import transformers as T

        torch.manual_seed(0)
        cfg = T.OlmoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=4, num_experts=16, num_experts_per_tok=4,
                            max_position_embeddings=2048, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model, name = T.OlmoeForCausalLM(cfg).to(device).eval(), "toy-olmoe"
        get = lambda d, n: toy_chunks(d, n, 256)  # noqa: E731
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device_map or device).eval()
        name = args.model.split("/")[-1]
        get = lambda d, n: chunks_for(tok, d, n, model.config.vocab_size)  # noqa: E731
    in_device = model.get_input_embeddings().weight.device
    spec = moescope.routing_spec(model.config)
    E = spec.num_experts

    t0 = time.time()
    target = get(args.target, args.plan_tokens + args.eval_tokens)
    n_eval = max(1, args.eval_tokens // SEQ)
    plan_t, eval_t = target[:-n_eval], target[-n_eval:]
    plan_o = get(args.other, args.plan_tokens)
    print(f"data: {len(plan_t)} plan + {len(eval_t)} held-out {args.target} chunks, "
          f"{len(plan_o)} {args.other} chunks ({time.time() - t0:.0f}s)", flush=True)

    layer_ids, mass_t, _ = log_usage(model, plan_t, spec, args.batch, in_device)
    _, mass_o, _ = log_usage(model, plan_o, spec, args.batch, in_device)
    _, _, held_counts = log_usage(model, eval_t, spec, args.batch, in_device)
    print(f"planning logs done ({time.time() - t0:.0f}s)", flush=True)

    restrict = Restrict(model, spec)
    rng = np.random.default_rng(0)
    full = perplexity(model, eval_t, args.batch, in_device)
    rows = [{"plan": "full model", "coverage": 1.0, "experts_per_layer": float(E), "ppl": full}]
    print(f"full model: ppl {full:.3f}", flush=True)
    for cov in (0.999, 0.99, 0.95, 0.9):
        own = experts_for_coverage(mass_t, cov)
        sizes = [len(s) for s in own]
        order_o = [np.argsort(-m, kind="stable") for m in mass_o]
        plans = {
            f"{args.target} log": own,
            f"{args.other} log": [np.sort(o[:n]) for o, n in zip(order_o, sizes)],
            "random": [np.sort(rng.choice(E, n, replace=False)) for n in sizes],
        }
        for label, kept in plans.items():
            restrict.allowed = dict(zip(layer_ids, kept))
            ppl = perplexity(model, eval_t, args.batch, in_device)
            rows.append({"plan": label, "coverage": cov, "experts_per_layer": float(np.mean(sizes)), "ppl": ppl})
            print(f"{label:>14} @ {cov:.3f}: {np.mean(sizes):5.1f}/{E} experts per layer, ppl {ppl:.3f} "
                  f"({100 * (ppl / full - 1):+.1f}%)", flush=True)
    restrict.allowed = None
    restrict.remove()

    result = {
        "model": name, "target": args.target, "other": args.other, "experts": E, "k": spec.k,
        "plan_tokens": int(plan_t.numel()), "eval_tokens": int(eval_t.numel()),
        "coverage_table": {d: {str(c): v for c, v in t.items()} for d, t in
                           coverage_table({args.target: mass_t, args.other: mass_o}).items()},
        "pruning": rows,
        "resident_hit_rate_own_plan": budget_curve(held_counts, mass_t).tolist(),
        "resident_hit_rate_other_plan": budget_curve(held_counts, mass_o).tolist(),
    }
    with open(os.path.join(args.out, "budget.json"), "w") as f:
        json.dump(result, f, indent=1)
    plot(result, os.path.join(args.out, "budget.png"))
    print(json.dumps({k: v for k, v in result.items() if "hit_rate" not in k}, indent=1))


def plot(r, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    full = r["pruning"][0]["ppl"]
    for label, color in ((f"{r['target']} log", "#4c72b0"), (f"{r['other']} log", "#dd8452"), ("random", "#8c8c8c")):
        pts = sorted((x["experts_per_layer"], 100 * (x["ppl"] / full - 1)) for x in r["pruning"] if x["plan"] == label)
        ax[0].plot(*zip(*pts), marker="o", color=color, label=f"experts kept from {label}")
    ax[0].set(title=f"{r['model']}: held-out {r['target']} perplexity after pruning", xlabel="experts kept per layer",
              ylabel="perplexity change (%)", yscale="symlog")
    ax[0].legend(fontsize=8)
    b = np.arange(len(r["resident_hit_rate_own_plan"]))
    ax[1].plot(b, 100 * np.array(r["resident_hit_rate_own_plan"]), color="#4c72b0", label=f"busiest on {r['target']} log")
    ax[1].plot(b, 100 * np.array(r["resident_hit_rate_other_plan"]), color="#dd8452", label=f"busiest on {r['other']} log")
    ax[1].plot(b, 100 * b / b[-1], color="#8c8c8c", ls="--", label="no plan (uniform)")
    ax[1].set(title=f"Held-out {r['target']} routing served by GPU-resident experts", xlabel="experts kept on GPU per layer",
              ylabel="decisions served (%)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
