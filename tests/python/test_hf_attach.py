"""moescope.attach() against the real Hugging Face routers.

Each case builds a tiny randomly initialised model of one MoE family (the
real transformers modelling code, no download), attaches moescope, runs a
forward pass, and checks that every (layer, token) expert set and its gates
match what the model's own router produced.

    python -m pytest -q tests/python/test_hf_attach.py
    MOESCOPE_DEVICE=cuda python -m pytest -q tests/python/test_hf_attach.py   # GPU kernel
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import moescope  # noqa: E402

DEVICE = os.environ.get("MOESCOPE_DEVICE", "cpu")
COMMON = dict(vocab_size=256, hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
              max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)


def _deepseek_like(cfg_cls, model_cls, **kw):
    cfg = cfg_cls(**COMMON, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=3,
                  first_k_dense_replace=1, n_shared_experts=1, q_lora_rank=None, kv_lora_rank=16,
                  qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16, **kw)
    model = model_cls(cfg)
    with torch.no_grad():  # a non-zero selection bias, so the bias path matters
        for mod in model.modules():
            if hasattr(mod, "e_score_correction_bias"):
                mod.e_score_correction_bias.normal_(0, 0.3)
    return model


def build(family):
    T = transformers
    torch.manual_seed(0)
    if family == "olmoe":
        return T.OlmoeForCausalLM(T.OlmoeConfig(**COMMON, intermediate_size=64, num_hidden_layers=2,
                                                num_experts=16, num_experts_per_tok=4))
    if family == "mixtral":
        return T.MixtralForCausalLM(T.MixtralConfig(**COMMON, intermediate_size=64, num_hidden_layers=2,
                                                    num_local_experts=8, num_experts_per_tok=2))
    if family == "granitemoe":
        return T.GraniteMoeForCausalLM(T.GraniteMoeConfig(**COMMON, intermediate_size=64, num_hidden_layers=2,
                                                          num_local_experts=16, num_experts_per_tok=4))
    if family in ("qwen3_moe", "qwen3_moe_nonorm"):
        return T.Qwen3MoeForCausalLM(T.Qwen3MoeConfig(**COMMON, intermediate_size=64, moe_intermediate_size=32,
                                                      num_hidden_layers=2, num_experts=16, num_experts_per_tok=4,
                                                      norm_topk_prob=family == "qwen3_moe", head_dim=16))
    if family == "qwen2_moe":
        return T.Qwen2MoeForCausalLM(T.Qwen2MoeConfig(**COMMON, intermediate_size=64, moe_intermediate_size=32,
                                                      shared_expert_intermediate_size=32, num_hidden_layers=2,
                                                      num_experts=16, num_experts_per_tok=4))
    if family == "deepseek_v3":
        return _deepseek_like(T.DeepseekV3Config, T.DeepseekV3ForCausalLM, n_routed_experts=32,
                              num_experts_per_tok=4, n_group=4, topk_group=2, norm_topk_prob=True,
                              routed_scaling_factor=2.5)
    if family == "deepseek_v3_nogroup":
        return _deepseek_like(T.DeepseekV3Config, T.DeepseekV3ForCausalLM, n_routed_experts=16,
                              num_experts_per_tok=4, n_group=1, topk_group=1, norm_topk_prob=False,
                              routed_scaling_factor=1.0)
    if family == "glm4_moe":
        cfg = T.Glm4MoeConfig(**COMMON, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=3,
                              first_k_dense_replace=1, n_shared_experts=1, n_routed_experts=16,
                              num_experts_per_tok=4, n_group=2, topk_group=1, norm_topk_prob=True,
                              routed_scaling_factor=1.8, head_dim=16)
        model = T.Glm4MoeForCausalLM(cfg)
        with torch.no_grad():
            for mod in model.modules():
                if hasattr(mod, "e_score_correction_bias"):
                    mod.e_score_correction_bias.normal_(0, 0.3)
        return model
    if family == "deepseek_v2":
        return _deepseek_like(T.DeepseekV2Config, T.DeepseekV2ForCausalLM, n_routed_experts=16,
                              num_experts_per_tok=4, topk_method="greedy", routed_scaling_factor=1.5)
    if family == "lfm2_moe":
        cfg = T.Lfm2MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                              num_hidden_layers=4, num_dense_layers=1, num_attention_heads=4, num_key_value_heads=4,
                              num_experts=16, num_experts_per_tok=4, norm_topk_prob=True, use_expert_bias=True,
                              routed_scaling_factor=1.5, layer_types=["full_attention", "conv", "full_attention", "conv"],
                              max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = T.Lfm2MoeForCausalLM(cfg)
        with torch.no_grad():
            for mod in model.modules():
                if hasattr(mod, "expert_bias"):
                    mod.expert_bias.normal_(0, 0.3)
        return model
    raise ValueError(family)


FAMILIES = ["olmoe", "mixtral", "granitemoe", "qwen3_moe", "qwen3_moe_nonorm", "qwen2_moe",
            "deepseek_v3", "deepseek_v3_nogroup", "glm4_moe", "deepseek_v2", "lfm2_moe"]


@pytest.mark.parametrize("family", FAMILIES)
def test_attach_matches_the_models_own_router(family):
    if DEVICE.startswith("cuda") and not moescope.cuda_available():
        pytest.skip("no CUDA")
    model = build(family).to(DEVICE if DEVICE != "cpu" else "cpu").eval()
    ids = torch.randint(3, 256, (2, 48), device=model.device)
    with moescope.attach(model, check=True, device=DEVICE) as session:
        with torch.no_grad():
            model(input_ids=ids)
            model(input_ids=ids[:, :17])  # second pass: token ids continue
    log, dropped = session.drain_with_stats()
    spec = session.spec
    n_layers = len(session.routers)
    assert dropped == 0
    assert len(log) == (2 * 48 + 2 * 17) * n_layers * spec.k
    assert log["token_id"].max() == 2 * 48 + 2 * 17 - 1
    par = session.parity(log)
    assert par is not None, "router did not expose its chosen experts"
    assert par["match_rate"] == 1.0, (family, par)
    assert par["max_gate_err"] < 1e-5, (family, par)


def test_deepseek_bias_and_groups_change_the_choice():
    """The test above is not vacuous: logging DeepSeek-V3 without its groups
    and with a zero bias must disagree with the model for some tokens."""
    import dataclasses

    model = build("deepseek_v3").eval()
    plain = dataclasses.replace(moescope.routing_spec(model.config), n_group=1, topk_group=1)
    ids = torch.randint(3, 256, (2, 48))
    with moescope.attach(model, check=True, device="cpu", spec=plain) as session:
        for layer_id in session.routers:
            session.recorder.set_bias(layer_id, np.zeros(plain.num_experts, np.float32))
        with torch.no_grad():
            model(input_ids=ids)
    assert session.parity(session.drain())["match_rate"] < 0.9


def test_unsupported_routing_is_refused():
    cfg = transformers.DeepseekV2Config(**COMMON, num_hidden_layers=1, n_routed_experts=8, num_experts_per_tok=2,
                                        topk_method="group_limited_greedy", n_group=2, topk_group=1)
    with pytest.raises(moescope.hf.UnsupportedRouting):
        moescope.routing_spec(cfg)


# -- capacity-factor models: tokens past an expert's capacity are dropped ----

def build_capacity(family, capacity=4):
    T = transformers
    torch.manual_seed(0)
    if family == "switch_transformers":
        cfg = T.SwitchTransformersConfig(vocab_size=256, d_model=32, d_kv=8, d_ff=64, num_heads=4, num_layers=2,
                                         num_decoder_layers=2, num_experts=4, expert_capacity=capacity,
                                         encoder_sparse_step=1, decoder_sparse_step=1, decoder_start_token_id=0)
        return T.SwitchTransformersForConditionalGeneration(cfg)
    if family == "nllb_moe":
        cfg = T.NllbMoeConfig(vocab_size=256, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
                              decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, num_experts=4,
                              expert_capacity=capacity, moe_eval_capacity_token_fraction=0.0, encoder_sparse_step=1,
                              decoder_sparse_step=1, max_position_embeddings=128, pad_token_id=1)
        return T.NllbMoeForConditionalGeneration(cfg)
    raise ValueError(family)


def _run_capacity(family, device="cpu"):
    model = build_capacity(family).to(device).eval()
    src = torch.randint(3, 256, (2, 24), device=model.device)
    tgt = torch.randint(3, 256, (2, 5), device=model.device)
    with moescope.attach(model, check=True, device=device) as session:
        with torch.no_grad():
            model(input_ids=src, decoder_input_ids=tgt)
    log, dropped = session.drain_with_stats()
    return session, log, dropped


@pytest.mark.parametrize("family", ["switch_transformers", "nllb_moe"])
def test_capacity_drops_are_flagged_from_the_routers_dispatch_mask(family):
    if DEVICE.startswith("cuda") and not moescope.cuda_available():
        pytest.skip("no CUDA")
    session, log, dropped = _run_capacity(family, DEVICE)
    spec = session.spec
    assert dropped == 0
    n_enc, n_dec = 2 * 24, 2 * 5
    enc_layers = [l for l in session.routers if l < 2]
    dec_layers = [l for l in session.routers if l >= 2]
    assert enc_layers == [0, 1] and dec_layers == [2, 3], session.routers.keys()
    # Every routing decision is logged, dropped or not; encoder and decoder
    # passes get consecutive token ids.
    assert len(log) == (n_enc * len(enc_layers) + n_dec * len(dec_layers)) * spec.k
    assert set(log["token_id"][log["layer_id"] < 2].tolist()) == set(range(n_enc))
    assert set(log["token_id"][log["layer_id"] >= 2].tolist()) == set(range(n_enc, n_enc + n_dec))
    drops = (log["flags"] & moescope.CAPACITY_DROP) != 0
    assert drops.any(), "capacity 4 with 48 tokens on 4 experts must overflow"
    # Per (layer, expert), at most `capacity` tokens survive in the encoder
    # (Switch counts capacity per sequence, NLLB per batch).
    per_seq = family == "switch_transformers"
    kept = log[~drops & (log["layer_id"] < 2)]
    seq = kept["token_id"] // 24 if per_seq else np.zeros(len(kept), np.int64)
    _, counts = np.unique(np.stack([kept["layer_id"], kept["expert_id"], seq]), axis=1, return_counts=True)
    assert counts.max() <= 4


def test_switch_log_matches_what_the_experts_computed():
    """Switch Transformer: parity is checked against the rows each expert
    module really received, so a dropped token must be flagged and a kept
    one must have reached its expert."""
    if DEVICE.startswith("cuda") and not moescope.cuda_available():
        pytest.skip("no CUDA")
    session, log, _ = _run_capacity("switch_transformers", DEVICE)
    par = session.parity(log)
    assert par["source"] == "experts"
    assert par["capacity_drops"] > 0
    assert par["match_rate"] == 1.0, par
    assert par["max_gate_err"] < 1e-5, par


def test_nllb_moe_experts_disagree_with_the_router_in_transformers_5():
    """transformers 5.x NLLB-MoE one-hot encodes the router's (already
    one-hot) top-1 mask again, so experts 0 and 1 receive the tokens instead
    of the experts the router chose. moescope logs the router's decision; the
    experts-side parity check is what exposes the mismatch. If this starts
    failing, transformers has fixed it: flip the assertion."""
    if DEVICE.startswith("cuda") and not moescope.cuda_available():
        pytest.skip("no CUDA")
    session, log, _ = _run_capacity("nllb_moe", DEVICE)
    par = session.parity(log)
    assert par["source"] == "experts"
    assert par["match_rate"] < 0.5, par


def test_layer_ids_past_255_are_refused():
    with pytest.raises(OverflowError):
        moescope.route_and_log(np.zeros((1, 4), np.float32), k=1, layer_id=256,
                               recorder=moescope.Recorder(capacity=8, device="cpu"))


def test_attach_keeps_the_module_tree_acyclic():
    """The router must not register its own layer as a submodule: the
    model has to stay usable (train(), to(), state_dict()) while attached."""
    model = build("olmoe")
    n_params = len(model.state_dict())
    with moescope.attach(model, device="cpu"):
        model.train()
        model.eval()
        assert len(model.state_dict()) == n_params
