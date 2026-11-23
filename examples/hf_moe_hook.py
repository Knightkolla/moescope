"""Hook one MoE layer of a Hugging Face model, log its routing with moescope,
and check that moescope picks the same experts (and gates) as the model.

By default builds a *tiny randomly initialised* OLMoE or Mixtral from its HF
config (no download; the architecture and router code are the real HF ones).
Pass --pretrained allenai/OLMoE-1B-7B-0924 to use real weights (~14 GB fp16).

    python examples/hf_moe_hook.py --arch olmoe
    python examples/hf_moe_hook.py --arch mixtral --device cuda

Exit code 0 and a final "OK" line mean every token's expert set matched.
"""

import argparse
import sys

import numpy as np
import torch

import moescope


def build_model(arch: str, pretrained: str, device: str):
    if pretrained:
        from transformers import AutoModelForCausalLM

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        return AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=dtype).to(device).eval()
    torch.manual_seed(0)
    common = dict(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                  bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if arch == "olmoe":
        from transformers import OlmoeConfig, OlmoeForCausalLM

        return OlmoeForCausalLM(OlmoeConfig(num_experts=16, num_experts_per_tok=4, **common)).to(device).eval()
    from transformers import MixtralConfig, MixtralForCausalLM

    return MixtralForCausalLM(MixtralConfig(num_local_experts=8, num_experts_per_tok=2, **common)).to(device).eval()


def router_settings(config):
    """(k, moescope mode) matching the model's own router."""
    k = config.num_experts_per_tok
    # Mixtral always renormalizes over the top-k; OLMoE does iff norm_topk_prob.
    renorm = getattr(config, "norm_topk_prob", config.model_type == "mixtral")
    return k, ("topk_softmax" if renorm else "softmax_topk")


def model_decision(output, k, renorm):
    """(logits, expert indices, gates) as the model computed them.

    transformers >= 5: the gate module returns (logits, scores, indices).
    transformers 4.x: the gate is a Linear; the block does softmax + topk.
    """
    if isinstance(output, tuple):
        logits, scores, indices = output[:3]
        return logits, indices, scores
    logits = output
    probs = torch.softmax(logits.float(), dim=-1)
    scores, indices = torch.topk(probs, k, dim=-1)
    if renorm:
        scores = scores / scores.sum(dim=-1, keepdim=True)
    return logits, indices, scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["olmoe", "mixtral"], default="olmoe")
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    args = ap.parse_args()

    model = build_model(args.arch, args.pretrained, args.device)
    k, mode = router_settings(model.config)
    backend = "cuda" if args.device.startswith("cuda") and moescope.cuda_available() else "cpu"
    rec = moescope.configure(capacity=1 << 22, mode=mode, device=backend)
    print(f"model={model.config.model_type} layer={args.layer} k={k} mode={mode} moescope backend={rec.backend}")

    gate = model.model.layers[args.layer].mlp.gate
    tokens_per_step = args.batch * args.seq
    state = {"step": 0, "reference": []}

    def hook(_module, _inputs, output):
        logits, indices, scores = model_decision(output, k, mode == "topk_softmax")
        offset = state["step"] * tokens_per_step
        moescope.route_and_log(logits, k=k, layer_id=args.layer, token_offset=offset)
        # Verification only (not part of normal telemetry use): keep the
        # model's own decision to compare against after the drain.
        state["reference"].append((offset, indices.reshape(-1, k).cpu(), scores.reshape(-1, k).float().cpu()))

    handle = gate.register_forward_hook(hook)
    with torch.no_grad():
        for step in range(args.steps):
            state["step"] = step
            ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq), device=args.device)
            model(input_ids=ids)
    handle.remove()

    records, dropped = moescope.drain_with_stats()
    n_tokens = args.steps * tokens_per_step
    assert dropped == 0, f"{dropped} records dropped"
    assert len(records) == n_tokens * k, (len(records), n_tokens * k)

    # Rebuild [token, k] tables from the log, ordered by expert id per token.
    records = records[np.lexsort((records["expert_id"], records["token_id"]))]
    ours_ids = records["expert_id"].reshape(n_tokens, k).astype(np.int64)
    ours_gate = records["gate"].reshape(n_tokens, k)
    ref_ids = torch.cat([r[1] for r in state["reference"]]).numpy()
    ref_gate = torch.cat([r[2] for r in state["reference"]]).numpy()
    order = np.argsort(ref_ids, axis=1)
    ref_ids, ref_gate = np.take_along_axis(ref_ids, order, 1), np.take_along_axis(ref_gate, order, 1)

    mismatched = np.flatnonzero((ours_ids != ref_ids).any(axis=1))
    gate_err = float(np.abs(ours_gate - ref_gate).max())
    # Half-precision models compute gates in fp16/bf16; float32 toy models agree to ~1e-6.
    gate_tol = 1e-5 if model.dtype == torch.float32 else 2e-3
    load = np.bincount(records["expert_id"], minlength=int(records["expert_id"].max()) + 1)
    print(f"logged {len(records)} decisions for {n_tokens} tokens; expert load = {load.tolist()}")
    if len(mismatched):
        t = mismatched[0]
        print(f"FAIL: {len(mismatched)} tokens differ, e.g. token {t}: moescope {ours_ids[t]} vs model {ref_ids[t]}")
        return 1
    if gate_err > gate_tol:
        print(f"FAIL: expert ids match but max gate diff {gate_err:.2e} > {gate_tol}")
        return 1
    print(f"OK: all {n_tokens} tokens x {k} experts match the model's router (max gate diff {gate_err:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
