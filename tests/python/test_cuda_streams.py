"""Serving-style use on the GPU: half-precision logits read natively, launches
on the caller's stream with no host sync, and logging captured into a CUDA
graph that is replayed with fresh token ids.

    MOESCOPE_DEVICE=cuda python -m pytest -q tests/python/test_cuda_streams.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import moescope  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and moescope.cuda_available()), reason="needs CUDA"
)

E, K = 64, 8


def by_token(log, n, k):
    srt = log[np.argsort(log["token_id"], kind="stable")]
    return srt["token_id"].reshape(n, k)[:, 0], srt["expert_id"].reshape(n, k).astype(np.int64), srt["gate"].reshape(n, k)


def same_choice(x, ids):
    """True per token if the logged experts have the same logits as torch's
    top-k (identical except on exact ties, where either choice is right)."""
    xf = x.float()
    ref = xf.topk(K, -1).indices
    mine = torch.from_numpy(ids).to(x.device)
    return (xf.gather(1, mine) == xf.gather(1, ref)).all(1).cpu().numpy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_half_precision_logits_are_read_natively(dtype):
    n = 4096
    x = (torch.randn(n, E, device="cuda") * 3).to(dtype)
    rec = moescope.Recorder(capacity=n * K, mode="topk_softmax", device="cuda")
    moescope.route_and_log(x, k=K, layer_id=0, recorder=rec)
    log, dropped = moescope.drain_with_stats(rec)
    assert dropped == 0 and len(log) == n * K
    _, ids, gates = by_token(log, n, K)
    assert same_choice(x, ids).all()
    ref_gates = torch.softmax(x.float().gather(1, torch.from_numpy(ids).cuda()), -1).cpu().numpy()
    assert np.abs(gates - ref_gates).max() < 1e-5


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64, torch.float16])
def test_kept_mask_on_device_sets_capacity_drop(mask_dtype):
    n = 3000
    x = torch.randn(n, E, device="cuda")
    kept = torch.rand(n, E, device="cuda") < 0.7
    rec = moescope.Recorder(capacity=n * K, mode="softmax_topk", device="cuda")
    moescope.route_and_log(x, k=K, layer_id=7, recorder=rec, kept=kept.to(mask_dtype))
    log, _ = moescope.drain_with_stats(rec)
    want = ~kept.cpu().numpy()[log["token_id"].astype(np.int64), log["expert_id"].astype(np.int64)]
    got = (log["flags"] & moescope.CAPACITY_DROP) != 0
    assert len(log) == n * K and np.array_equal(got, want) and want.any()


def test_sorted_drain_is_identical_across_runs():
    """Append order on the GPU depends on which thread group wins the atomic;
    sort=True must give the same bytes every run, with each token's k
    records in rank order (best first)."""
    n = 50_000
    x = torch.randn(n, E, device="cuda")
    runs = []
    for _ in range(3):
        rec = moescope.Recorder(capacity=2 * n * K, mode="softmax_topk", device="cuda")
        for layer in (1, 0):
            moescope.route_and_log(x, k=K, layer_id=layer, recorder=rec)
        runs.append(moescope.drain(rec, sort=True))
    assert all(r.tobytes() == runs[0].tobytes() for r in runs[1:])
    log = runs[0]
    assert (np.diff(log["layer_id"].astype(np.int64)) >= 0).all()
    g = log["gate"].reshape(-1, K)
    assert (np.diff(g, axis=1) <= 0).all(), "within a token, best expert first"
    t = log["token_id"].reshape(-1, K)
    assert (t == t[:, :1]).all() and np.array_equal(t[: n, 0], np.arange(n))


def test_logits_from_a_non_blocking_stream_need_no_sync():
    n = 8192
    rec = moescope.Recorder(capacity=n * K, mode="softmax_topk", device="cuda")
    side = torch.cuda.Stream()  # non-blocking w.r.t. the legacy default stream
    with torch.cuda.stream(side):
        torch.cuda._sleep(100_000_000)  # producer is slow...
        x = torch.randn(n, E, device="cuda") + 0
        moescope.route_and_log(x, k=K, layer_id=0, recorder=rec)  # ...and we do not wait for it
    log, dropped = moescope.drain_with_stats(rec)  # drain syncs the whole context
    _, ids, _ = by_token(log, n, K)
    assert dropped == 0 and same_choice(x, ids).all()


def test_cuda_graph_replays_log_fresh_token_ids():
    n, replays = 1024, 5
    rec = moescope.Recorder(capacity=n * K * (replays + 1), mode="softmax_topk", device="cuda")
    rec.set_device_token_ids(True)
    static = torch.randn(n, E, device="cuda")

    # Warm up on a side stream (torch's rule before capture), then discard it.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        moescope.route_and_log(static, k=K, layer_id=3, recorder=rec)
    torch.cuda.current_stream().wait_stream(s)
    moescope.drain_with_stats(rec)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moescope.route_and_log(static, k=K, layer_id=3, recorder=rec)
        moescope.advance_tokens(n, recorder=rec)

    data = [torch.randn(n, E, device="cuda") for _ in range(replays)]
    for d in data:
        static.copy_(d)
        graph.replay()
    log, dropped = moescope.drain_with_stats(rec)
    assert dropped == 0 and len(log) == replays * n * K
    assert (log["layer_id"] == 3).all()
    tokens, ids, _ = by_token(log, replays * n, K)
    assert np.array_equal(tokens, np.arange(replays * n)), "each replay must get new token ids"
    for r, d in enumerate(data):
        assert same_choice(d, ids[r * n : (r + 1) * n]).all(), f"replay {r} logged the wrong experts"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_model_split_over_two_gpus_logs_without_host_sync(monkeypatch):
    """device_map-style split: layer 1's router runs on cuda:1, the log lives
    on cuda:0. The hook must not synchronize, and the log must equal a
    checked run's."""
    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    torch.manual_seed(0)
    cfg = transformers.OlmoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, num_experts=16,
                                   num_experts_per_tok=4, max_position_embeddings=128, pad_token_id=0)
    model = transformers.OlmoeForCausalLM(cfg).eval()
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.layers.0": 0,
                  "model.layers.1": 1, "model.norm": 1, "lm_head": 1}
    model = accelerate.dispatch_model(model, device_map=device_map)
    ids = torch.randint(3, 256, (2, 64), device="cuda:0")

    def run(check):
        with moescope.attach(model, check=check, device="cuda:0") as s:
            with torch.no_grad():
                model(input_ids=ids)
        return s, s.drain()

    ref_session, ref = run(check=True)
    par = ref_session.parity(ref)
    assert par["match_rate"] == 1.0, par

    calls = []
    real = torch.cuda.Stream.synchronize
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append("device"))
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", lambda self: (calls.append("stream"), real(self)))
    with moescope.attach(model, device="cuda:0") as s:
        with torch.no_grad():
            model(input_ids=ids)
    assert calls == [], f"hook synchronized: {calls}"
    monkeypatch.undo()
    log = s.drain()
    key = lambda x: np.sort(x, order=["layer_id", "token_id", "expert_id"])  # noqa: E731
    a, b = key(log), key(ref)
    assert np.array_equal(a[["token_id", "layer_id", "expert_id"]], b[["token_id", "layer_id", "expert_id"]])
    assert np.abs(a["gate"] - b["gate"]).max() < 1e-6
