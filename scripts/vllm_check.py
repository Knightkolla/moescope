"""moescope.vllm against a real vLLM engine (GPU).

1. Serves Granite-3.1-1B-A400M with vLLM (enforce_eager), attaches moescope,
   generates for a few prompts, and checks every processed token was logged
   once per MoE layer x top-k.
2. Runs the same prompt through Hugging Face transformers with
   moescope.attach and compares the expert sets logged for the prompt
   tokens (fp16 on both sides, different kernels: near, not exactly, 100%).

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python scripts/vllm_check.py
"""

import json
import os
import sys

import numpy as np

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "ibm-granite/granite-3.1-1b-a400m-instruct"
PROMPTS = [
    "def quicksort(xs):\n    ",
    "The capital of France is",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
]


def expert_sets(log, k):
    """{(layer, token index within this log): expert ids}; token ids are
    rebased because the session's counter keeps running across drains."""
    log = log.copy()
    log["token_id"] -= log["token_id"].min()
    srt = np.sort(log, order=["layer_id", "token_id", "expert_id"])
    return {(int(r[0]["layer_id"]), int(r[0]["token_id"])): tuple(r["expert_id"].tolist())
            for r in srt.reshape(-1, k)}


def main():
    from vllm import LLM, SamplingParams

    import moescope.vllm

    llm = LLM(MODEL, enforce_eager=True, dtype="float16", gpu_memory_utilization=0.6, max_model_len=1024,
              enable_prefix_caching=False)
    session = moescope.vllm.attach(llm)
    spec = session.spec
    out = llm.generate(PROMPTS, SamplingParams(max_tokens=32, temperature=0))
    log, lost = session.drain_with_stats()
    n_layers = len(session.layers)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in out)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in out)
    tokens = len(np.unique(log["token_id"]))
    res = {"model": MODEL, "moe_layers": n_layers, "k": spec.k, "mode": spec.mode, "records": int(len(log)),
           "lost": int(lost), "tokens_logged": tokens, "prompt_tokens": prompt_tokens,
           "generated_tokens": gen_tokens}
    # every processed token (prompt + each decode step but the last sampled one) logged once per layer x k
    assert lost == 0 and len(log) == tokens * n_layers * spec.k, res
    assert tokens >= prompt_tokens + gen_tokens - len(PROMPTS), res

    # Prompt-only pass for the HF comparison: one prompt, prefill only.
    session.drain()
    one = llm.generate([PROMPTS[2]], SamplingParams(max_tokens=1, temperature=0))
    vlog = session.drain()
    ids = one[0].prompt_token_ids
    session.detach()
    del llm

    import torch
    from transformers import AutoModelForCausalLM

    import moescope

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    with moescope.attach(model) as s:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"))
    hlog = s.drain()
    a, b = expert_sets(vlog, spec.k), expert_sets(hlog, spec.k)
    common = set(a) & set(b)
    agree = np.mean([a[x] == b[x] for x in common])
    res.update({"hf_compared": len(common), "hf_same_expert_set": float(agree)})
    print(json.dumps(res, indent=1))
    assert len(common) == len(ids) * n_layers, (len(common), len(ids), n_layers)
    assert agree > 0.95, res
    print("VLLM CHECK PASSED")


if __name__ == "__main__":
    main()
