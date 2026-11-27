import os
import sys

import numpy as np
import pytest

import moescope
from moescope.analysis import budget_curve, expert_usage, experts_for_coverage, resident_hit_rate


def _log(rows):
    log = np.zeros(len(rows), moescope.RECORD_DTYPE)
    for i, (t, l, e, g, f) in enumerate(rows):
        log[i] = (t, l, f, e, g)
    return log


def test_usage_coverage_and_hit_rate():
    log = _log([(0, 2, 0, 0.5, 0), (1, 2, 0, 0.5, 0), (2, 2, 1, 0.5, 0), (3, 2, 3, 0.9, moescope.CAPACITY_DROP),
                (0, 5, 2, 1.0, 0)])
    ids, c = expert_usage(log, 4)
    assert ids == [2, 5] and c.tolist() == [[2, 1, 0, 0], [0, 0, 1, 0]]  # the capacity drop is not counted
    _, g = expert_usage(log, 4, "gate", include_capacity_drops=True)
    assert np.allclose(g[0], [1.0, 0.5, 0, 0.9])
    assert [s.tolist() for s in experts_for_coverage(c, 0.6)] == [[0], [2]]
    assert [s.tolist() for s in experts_for_coverage(c, 0.9)] == [[0, 1], [2]]
    assert resident_hit_rate(c, 1).tolist() == [2 / 3, 1.0]
    curve = budget_curve(c)
    assert curve[0] == 0 and curve[-1] == 1 and np.all(np.diff(curve) >= 0)


def test_restricted_routing_only_uses_allowed_experts():
    torch = pytest.importorskip("torch")
    T = pytest.importorskip("transformers")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
    from expert_budget import Restrict

    torch.manual_seed(0)
    cfg = T.OlmoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=4, num_experts=16, num_experts_per_tok=4,
                        max_position_embeddings=128, pad_token_id=0)
    model = T.OlmoeForCausalLM(cfg).eval()
    ids = torch.randint(3, 256, (2, 32))
    r = Restrict(model, moescope.routing_spec(cfg))
    allowed = {0: np.array([1, 3, 5, 7, 9]), 1: np.arange(8, 16)}
    r.allowed = allowed
    with moescope.attach(model, check=True, device="cpu") as s:
        with torch.no_grad():
            model(input_ids=ids)
    for layer, _off, computed, _w in s._reference:  # what the experts actually ran
        used = np.flatnonzero(computed.sum(0))
        assert set(used) <= set(allowed[layer]) and (computed.sum(1) == 4).all() and computed.max() == 1
    r.remove()
