"""moescope: MoE routing telemetry.

Route MoE tokens to experts and log every decision ``(token_id, layer_id,
expert_id, gate)``. With the CUDA backend the log lives in GPU memory and is
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
    "RECORD_DTYPE",
    "Recorder",
    "attach",
    "built_with_cuda",
    "configure",
    "cuda_available",
    "drain",
    "drain_with_stats",
    "route_and_log",
    "routing_spec",
    "set_bias",
]

#: One row per routing decision; byte-identical to the Rust/CUDA struct.
RECORD_DTYPE = np.dtype(
    [("token_id", "<u4"), ("layer_id", "<u2"), ("expert_id", "<u2"), ("gate", "<f4")]
)
assert RECORD_DTYPE.itemsize == _native.RECORD_ITEMSIZE == 12

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


def _is_torch_tensor(x: Any) -> bool:
    return type(x).__module__.startswith("torch") and hasattr(x, "data_ptr")


def route_and_log(
    logits: Any,
    k: int,
    layer_id: int,
    token_offset: int = 0,
    recorder: Optional[Recorder] = None,
) -> None:
    """Route one MoE layer's tokens and append each decision to the log.

    logits: router logits, ``[..., num_experts]`` (leading dims are
        flattened into tokens, row-major), as a numpy array or torch tensor.
        A CUDA tensor with the CUDA backend is read in place (zero-copy, no
        host sync); anything else is copied to host first.
    k: experts per token.
    layer_id: tag stored in every record (0-65535).
    token_offset: added to the flattened token index to form ``token_id``.
    """
    rec = _recorder(recorder)
    if _is_torch_tensor(logits):
        x = logits.detach()
        x = x.reshape(-1, x.shape[-1])
        if x.is_cuda and rec.backend == "cuda":
            import torch

            x = x.to(torch.float32).contiguous()
            # Our kernel runs on the legacy default stream; make sure logits
            # produced on any other stream are finished first.
            if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):
                torch.cuda.current_stream(x.device).synchronize()
            rec.route_and_log_device(x.data_ptr(), x.shape[0], x.shape[1], k, layer_id, token_offset)
            return
        logits = x.float().cpu().numpy()
    x = np.ascontiguousarray(logits, dtype=np.float32)
    x = x.reshape(-1, x.shape[-1])
    rec.route_and_log(x, k, layer_id, token_offset)


def drain_with_stats(recorder: Optional[Recorder] = None) -> Tuple[np.ndarray, int]:
    """Copy the whole log to host (one transfer) and reset it.

    Returns ``(records, dropped)``. On the GPU, record order is the order
    threads appended them, not token order; sort if you need one.
    """
    raw, dropped = _recorder(recorder).drain()
    return np.frombuffer(raw, dtype=RECORD_DTYPE).copy(), int(dropped)


def drain(recorder: Optional[Recorder] = None) -> np.ndarray:
    """Like :func:`drain_with_stats` but returns only the records, warning
    if any were dropped because the buffer was full."""
    records, dropped = drain_with_stats(recorder)
    if dropped:
        warnings.warn(
            f"moescope: {dropped} routing records dropped (capacity full); "
            "raise capacity or drain more often",
            RuntimeWarning,
            stacklevel=2,
        )
    return records


from .hf import attach, routing_spec  # noqa: E402  (needs the names above)
