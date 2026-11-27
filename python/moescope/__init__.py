"""moescope: MoE routing telemetry.

Route MoE tokens to experts and log every decision ``(token_id, layer_id,
flags, expert_id, gate)``. With the CUDA backend the log lives in GPU memory and is
written by the routing kernel itself, so nothing crosses PCIe until you call
:func:`drain` once at the end.

Typical use inside a forward hook on an MoE router::

    import moescope
    moescope.configure(capacity=1 << 24, mode="softmax_topk")  # OLMoE
    ...
    moescope.route_and_log(router_logits, k=8, layer_id=i)      # each step
    ...
    log = moescope.drain()   # numpy structured array, one row per decision
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np

from . import _native
from ._native import Recorder, built_with_cuda, cuda_available

__all__ = [
    "CAPACITY_DROP",
    "RECORD_DTYPE",
    "Recorder",
    "advance_tokens",
    "attach",
    "built_with_cuda",
    "configure",
    "cuda_available",
    "drain",
    "drain_with_stats",
    "route_and_log",
    "routing_spec",
    "sort_records",
    "set_bias",
]

#: One row per routing decision; byte-identical to the Rust/CUDA struct.
RECORD_DTYPE = np.dtype(
    [("token_id", "<u4"), ("layer_id", "u1"), ("flags", "u1"), ("expert_id", "<u2"), ("gate", "<f4")]
)
assert RECORD_DTYPE.itemsize == _native.RECORD_ITEMSIZE == 12

#: ``flags`` bit: the router chose this expert, but the expert never computed
#: the token because it was full (capacity-factor MoE such as Switch
#: Transformer or NLLB-MoE). ``gate`` is still the router's gate. Set only
#: when the model's dispatch mask is passed as ``kept`` (``attach`` does).
CAPACITY_DROP = int(_native.CAPACITY_DROP)

_default: Optional[Recorder] = None


def configure(
    capacity: int = 1 << 22,
    mode: str = "softmax_topk",
    device: str = "auto",
    *,
    normalize: bool = False,
    scale: float = 1.0,
    n_group: int = 1,
    topk_group: int = 1,
) -> Recorder:
    """(Re)create the default recorder.

    capacity: max records held between drains (12 bytes each). Records past
        this are counted as dropped, never silently lost.
    mode: "softmax_topk" (softmax over all experts, keep top-k probs; OLMoE,
        Qwen-MoE without norm_topk_prob), "topk_softmax" (softmax over the
        selected k; Mixtral, Granite) or "sigmoid" (DeepSeek-V3, GLM-4.5,
        Kimi K2, Llama 4).
    device: "auto", "cpu", "cuda" or "cuda:N".
    normalize: divide the k gates by their sum (``norm_topk_prob``).
    scale: multiply the gates (``routed_scaling_factor``).
    n_group, topk_group: DeepSeek-V3 group-limited routing (sigmoid only).

    For Hugging Face models, :func:`attach` sets all of this from the config.
    """
    global _default
    _default = Recorder(
        capacity=capacity,
        mode=mode,
        device=device,
        normalize=normalize,
        scale=scale,
        n_group=n_group,
        topk_group=topk_group,
    )
    return _default


def set_bias(layer_id: int, bias: Any, recorder: Optional[Recorder] = None) -> None:
    """Sets layer ``layer_id``'s per-expert selection bias (sigmoid mode),
    e.g. DeepSeek-V3's ``e_score_correction_bias``. Copied once; reused for
    every later call on that layer."""
    if _is_torch_tensor(bias):
        bias = bias.detach().float().cpu().numpy()
    _recorder(recorder).set_bias(layer_id, np.ascontiguousarray(bias, dtype=np.float32))


def _recorder(recorder: Optional[Recorder]) -> Recorder:
    if recorder is not None:
        return recorder
    return _default if _default is not None else configure()


def _torch_dtypes():
    try:
        import torch
    except ImportError:
        return {}
    return {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}


_TORCH_DTYPES = _torch_dtypes()


def _is_torch_tensor(x: Any) -> bool:
    return type(x).__module__.startswith("torch") and hasattr(x, "data_ptr")


def route_and_log(
    logits: Any,
    k: int,
    layer_id: int,
    token_offset: int = 0,
    recorder: Optional[Recorder] = None,
    kept: Any = None,
) -> None:
    """Route one MoE layer's tokens and append each decision to the log.

    logits: router logits, ``[..., num_experts]`` (leading dims are
        flattened into tokens, row-major), as a numpy array or torch tensor.
        A CUDA tensor (float32 / float16 / bfloat16) with the CUDA backend is
        read in place on the current stream (zero-copy, no host sync, CUDA
        graph capturable); anything else is copied to host first.
    k: experts per token.
    layer_id: tag stored in every record (0-255).
    token_offset: added to the flattened token index to form ``token_id``.
    kept: optional ``[..., num_experts]`` dispatch mask after capacity, same
        leading shape as ``logits`` (nonzero = the expert computed the
        token). Decisions it marks 0 get ``flags & CAPACITY_DROP``.
    """
    rec = _recorder(recorder)
    if _is_torch_tensor(logits):
        x = logits.detach()
        x = x.reshape(-1, x.shape[-1])
        if x.is_cuda and rec.backend == "cuda":
            import torch

            dtype = _TORCH_DTYPES.get(x.dtype)
            if dtype is None:
                x, dtype = x.float(), "float32"
            x = x.contiguous()
            m = None
            if kept is not None:
                m = _as_torch(kept, x.device).reshape(x.shape)
                m = (m != 0).to(torch.uint8).contiguous()
            # Launch on the caller's current stream: ordered after the op that
            # produced the logits, no host sync, and capturable in a CUDA graph.
            # (The temporaries above are freed in that stream's order too.)
            stream = torch.cuda.current_stream(x.device).cuda_stream
            rec.route_and_log_device(
                x.data_ptr(), x.shape[0], x.shape[1], k, layer_id, token_offset, dtype=dtype, stream=stream,
                kept_ptr=0 if m is None else m.data_ptr(),
            )
            return
        logits = x.float().cpu().numpy()
    x = np.ascontiguousarray(logits, dtype=np.float32)
    x = x.reshape(-1, x.shape[-1])
    m = None
    if kept is not None:
        if _is_torch_tensor(kept):
            kept = kept.detach().cpu().numpy()
        m = np.ascontiguousarray(np.asarray(kept).reshape(x.shape) != 0, dtype=np.uint8)
    rec.route_and_log(x, k, layer_id, token_offset, kept=m)


def _as_torch(t: Any, device):
    import torch

    t = t.detach() if _is_torch_tensor(t) else torch.as_tensor(np.asarray(t))
    return t.to(device)


def advance_tokens(n: int, recorder: Optional[Recorder] = None) -> None:
    """With device-side token ids (``Recorder.set_device_token_ids(True)``),
    moves the token counter past a forward pass of ``n`` tokens, on the
    current CUDA stream. Inside a captured CUDA graph this is recorded too,
    so each replay logs new token ids."""
    rec = _recorder(recorder)
    if rec.backend != "cuda":
        return
    import torch

    rec.advance_tokens(n, stream=torch.cuda.current_stream().cuda_stream)


def sort_records(records: np.ndarray) -> np.ndarray:
    """Canonical order: by ``layer_id``, then ``token_id``; within a token,
    the router's rank order (best expert first) is kept, since the sort is
    stable and each token's k records are written contiguously in rank
    order. Two runs over the same inputs then compare row by row."""
    return records[np.lexsort((records["token_id"], records["layer_id"]))]


def drain_with_stats(recorder: Optional[Recorder] = None, sort: bool = False) -> Tuple[np.ndarray, int]:
    """Copy the whole log to host (one transfer) and reset it.

    Returns ``(records, dropped)``. On the GPU the k records of one token
    are contiguous and in rank order, but the order of tokens is the order
    their thread groups won the append atomic, so it changes from run to
    run. ``sort=True`` returns :func:`sort_records` order instead (costs a
    host sort, a few seconds for ~50M records).
    """
    raw, dropped = _recorder(recorder).drain()
    records = np.frombuffer(raw, dtype=RECORD_DTYPE).copy()
    return (sort_records(records) if sort else records), int(dropped)


def drain(recorder: Optional[Recorder] = None, sort: bool = False) -> np.ndarray:
    """Like :func:`drain_with_stats` but returns only the records, warning
    if any were dropped because the buffer was full."""
    records, dropped = drain_with_stats(recorder, sort=sort)
    if dropped:
        warnings.warn(
            f"moescope: {dropped} routing records dropped (capacity full); "
            "raise capacity or drain more often",
            RuntimeWarning,
            stacklevel=2,
        )
    return records


from .hf import attach, routing_spec  # noqa: E402  (needs the names above)
