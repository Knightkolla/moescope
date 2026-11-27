"""moescope.vllm without vLLM: a stand-in LLM whose apply_model runs the
function in-process on a toy model shaped like vLLM's (decoder layers whose
MoE module takes forward(hidden_states, router_logits))."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import moescope  # noqa: E402
import moescope.vllm as mv  # noqa: E402

E, K, H = 16, 4, 32


class Runner(torch.nn.Module):  # vLLM's fused MoE: takes the logits
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(H, H)

    def forward(self, hidden_states, router_logits):
        return self.w(hidden_states)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(H, E, bias=False)
        self.experts = Runner()

    def forward(self, x):
        return self.experts(hidden_states=x, router_logits=self.gate(x))


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Block()

    def forward(self, x):
        return x + self.mlp(x)


class Model(torch.nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(Layer() for _ in range(n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeLLM:
    def __init__(self, model, hf_config, eager=True):
        self.model = model
        cfg = type("MC", (), {"hf_config": hf_config, "enforce_eager": eager})()
        self.llm_engine = type("Eng", (), {"model_config": cfg})()

    def apply_model(self, fn):
        return [fn(self.model)]


def test_vllm_session_logs_every_layer_and_pass():
    torch.manual_seed(0)
    model = Model().eval()
    cfg = transformers.OlmoeConfig(num_experts=E, num_experts_per_tok=K)
    llm = FakeLLM(model, cfg)
    seen = {}
    for i, layer in enumerate(model.layers):
        layer.mlp.gate.register_forward_hook(lambda m, a, o, i=i: seen.setdefault(i, []).append(o.detach().clone()))
    with mv.attach(llm, device="cpu") as s:
        assert s.layers == [0, 1, 2]
        with torch.no_grad():
            model(torch.randn(10, H))
            model(torch.randn(4, H))
        log = s.drain()
    assert len(log) == (10 + 4) * 3 * K
    ref = moescope.Recorder(capacity=len(log), mode="softmax_topk", device="cpu")
    for i, outs in seen.items():
        moescope.route_and_log(outs[0], k=K, layer_id=i, token_offset=0, recorder=ref)
        moescope.route_and_log(outs[1], k=K, layer_id=i, token_offset=10, recorder=ref)
    want = moescope.drain(ref)
    key = ["layer_id", "token_id", "expert_id"]
    assert np.array_equal(np.sort(log, order=key), np.sort(want, order=key))


def test_vllm_needs_eager():
    llm = FakeLLM(Model(), transformers.OlmoeConfig(num_experts=E, num_experts_per_tok=K), eager=False)
    with pytest.raises(moescope.hf.UnsupportedRouting):
        mv.attach(llm, device="cpu")
