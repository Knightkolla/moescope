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
