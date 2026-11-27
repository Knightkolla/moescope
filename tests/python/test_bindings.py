"""Phase 5: Python bindings. Runs on whichever backend is available
(CPU locally; set MOESCOPE_DEVICE=cuda on Colab to force the GPU path)."""

import os
import warnings

import numpy as np
import pytest

import moescope

DEVICE = os.environ.get("MOESCOPE_DEVICE", "auto")


def reference(logits, k, mode):
    """Plain numpy router: top-k on logits (ties -> lower index), gates per mode."""
    order = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    if mode == "softmax_topk":
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = z / z.sum(axis=1, keepdims=True)
        gates = np.take_along_axis(probs, order, axis=1)
    else:
        sel = np.take_along_axis(logits, order, axis=1)
        z = np.exp(sel - sel.max(axis=1, keepdims=True))
        gates = z / z.sum(axis=1, keepdims=True)
    return order, gates


def sort_records(r):
    return r[np.lexsort((r["expert_id"], r["token_id"], r["layer_id"]))]


@pytest.fixture
def rec():
    return moescope.Recorder(capacity=1 << 16, mode="softmax_topk", device=DEVICE)


def test_dtype_is_12_bytes():
    assert moescope.RECORD_DTYPE.itemsize == 12
    assert moescope.RECORD_DTYPE.names == ("token_id", "layer_id", "flags", "expert_id", "gate")


@pytest.mark.parametrize("mode", ["softmax_topk", "topk_softmax"])
def test_matches_numpy_reference(mode):
    rec = moescope.Recorder(capacity=1 << 16, mode=mode, device=DEVICE)
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((257, 16)).astype(np.float32)
    moescope.route_and_log(logits, k=4, layer_id=3, token_offset=10, recorder=rec)
    records, dropped = moescope.drain_with_stats(rec)
    assert dropped == 0 and len(records) == 257 * 4
    records = sort_records(records)
    ids, gates = reference(logits, 4, mode)
    # Within a token, sort reference by expert id to match the record sort.
    perm = np.argsort(ids, axis=1)
    np.testing.assert_array_equal(records["expert_id"].reshape(257, 4), np.take_along_axis(ids, perm, 1))
    np.testing.assert_allclose(records["gate"].reshape(257, 4), np.take_along_axis(gates, perm, 1), atol=1e-5)
    np.testing.assert_array_equal(np.unique(records["token_id"]), np.arange(10, 267))
    assert (records["layer_id"] == 3).all()


def test_accepts_3d_and_torch(rec):
    torch = pytest.importorskip("torch")
    x = torch.randn(2, 5, 8)  # [batch, seq, experts]
    moescope.route_and_log(x, k=2, layer_id=0, recorder=rec)
    moescope.route_and_log(x.to(torch.bfloat16), k=2, layer_id=1, recorder=rec)
    r = moescope.drain(rec)
    assert len(r) == 2 * 10 * 2
    assert set(r["token_id"]) == set(range(10))


def test_drain_resets(rec):
    moescope.route_and_log(np.zeros((3, 4), np.float32), k=1, layer_id=0, recorder=rec)
    assert len(moescope.drain(rec)) == 3
    assert len(moescope.drain(rec)) == 0


def test_overflow_is_reported():
    rec = moescope.Recorder(capacity=10, device=DEVICE)
    moescope.route_and_log(np.zeros((8, 4), np.float32), k=2, layer_id=0, recorder=rec)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = moescope.drain(rec)
    assert len(r) == 10
    assert any("6 routing records dropped" in str(x.message) for x in w)


@pytest.mark.parametrize(
    "logits,k,msg",
    [
        (np.zeros((2, 4), np.float32), 5, "exceeds num_experts"),
        (np.array([[0.0, np.nan]], np.float32), 1, "non-finite"),
        (np.zeros((2, 4), np.float32), 0, "k must be"),
    ],
)
def test_bad_input_raises_value_error(rec, logits, k, msg):
    with pytest.raises(ValueError, match=msg):
        moescope.route_and_log(logits, k=k, layer_id=0, recorder=rec)


def test_bad_mode_and_device():
    with pytest.raises(ValueError):
        moescope.Recorder(mode="nope")
    with pytest.raises(ValueError):
        moescope.Recorder(device="tpu")


@pytest.mark.skipif(not moescope.cuda_available(), reason="needs CUDA backend")
def test_zero_copy_cuda_tensor_matches_host_path():
    torch = pytest.importorskip("torch")
    gpu = moescope.Recorder(capacity=1 << 16, device="cuda")
    cpu = moescope.Recorder(capacity=1 << 16, device="cpu")
    x = torch.randn(1000, 64, device="cuda")
    moescope.route_and_log(x, k=8, layer_id=2, recorder=gpu)  # zero-copy path
    moescope.route_and_log(x.cpu(), k=8, layer_id=2, recorder=cpu)
    g, c = sort_records(moescope.drain(gpu)), sort_records(moescope.drain(cpu))
    np.testing.assert_array_equal(g[["token_id", "layer_id", "expert_id"]], c[["token_id", "layer_id", "expert_id"]])
    np.testing.assert_allclose(g["gate"], c["gate"], atol=1e-5)


@pytest.mark.skipif(not moescope.cuda_available(), reason="needs CUDA backend")
def test_bad_device_pointer_is_rejected_not_dereferenced():
    rec = moescope.Recorder(device="cuda")
    with pytest.raises(ValueError, match="invalid device pointer"):
        rec.route_and_log_device(0xDEAD0000, 4, 8, 2, 0)
    torch = pytest.importorskip("torch")
    small = torch.zeros(4, 8, device="cuda")
    # Validation is per driver allocation (torch pools tensors into larger
    # segments), so overrun far past any plausible segment: 64M x 8 floats.
    with pytest.raises(ValueError, match="OutOfRange"):
        rec.route_and_log_device(small.data_ptr(), 64 << 20, 8, 2, 0)


def test_kept_mask_flags_capacity_drops_on_cpu():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 16)).astype(np.float32)
    kept = rng.random((500, 16)) < 0.6
    rec = moescope.Recorder(capacity=500 * 4, device="cpu")
    moescope.route_and_log(x, k=4, layer_id=3, recorder=rec, kept=kept)
    log = moescope.drain(rec)
    want = ~kept[log["token_id"].astype(np.int64), log["expert_id"].astype(np.int64)]
    assert np.array_equal((log["flags"] & moescope.CAPACITY_DROP) != 0, want) and want.any()
    with pytest.raises(ValueError):
        moescope.route_and_log(x, k=4, layer_id=3, recorder=rec, kept=kept[:10].ravel())
