"""Routing telemetry for models served by vLLM.

    from vllm import LLM
    import moescope.vllm

    llm = LLM("ibm-granite/granite-3.1-1b-a400m-instruct", enforce_eager=True)
    session = moescope.vllm.attach(llm)
    llm.generate(prompts)
    log = session.drain()            # token_id, layer_id, flags, expert_id, gate

vLLM runs the model in a worker process, so `attach` installs the hooks
there (through ``LLM.apply_model``) and `drain` ships the log back. Each
hook sits on a MoE layer's ``forward(hidden_states, router_logits)``, the
one interface every vLLM MoE layer shares across releases, and makes one
kernel launch on those logits; nothing is recomputed in Python.

``token_id`` counts tokens in the order the engine processed them. A vLLM
step mixes tokens from many requests (prefill chunks and decode tokens), so
ids are per processed token, not per request position. Prefix-cache hits
are not recomputed, so they are not logged either.

Needs ``enforce_eager=True``: with torch.compile / CUDA graphs vLLM traces
the model with ``fullgraph=True`` and Python hooks cannot run inside it.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Optional

import numpy as np

from .hf import RoutingSpec, UnsupportedRouting, routing_spec

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_BIAS_NAMES = ("e_score_correction_bias", "expert_bias")

# Worker-process state, keyed by session id (one LLM can have one session).
_WORKER: Dict[str, "_WorkerSession"] = {}


def _takes_router_logits(mod) -> bool:
    try:
        return "router_logits" in inspect.signature(mod.forward).parameters
    except (TypeError, ValueError):
        return False


def find_moe_layers(model) -> Dict[int, Any]:
    """``{layer_index: module}``: the outermost module inside each decoder
    layer whose ``forward`` takes ``router_logits``."""
    found: Dict[int, Any] = {}
    names: Dict[int, str] = {}
    for name, mod in model.named_modules():
        m = _LAYER.search(name)
        if m is None or not _takes_router_logits(mod):
            continue
        i = int(m.group(1))
        if i in names and name.startswith(names[i] + "."):
            continue  # nested inside the one already found
        found[i], names[i] = mod, name
    return found


def _find_bias(mod, parent) -> Optional[Any]:
    for root in (mod, parent):
        if root is None:
            continue
        for name, t in list(root.named_parameters()) + list(root.named_buffers()):
            if name.rsplit(".", 1)[-1] in _BIAS_NAMES and t.dim() == 1:
                return t
        for attr in _BIAS_NAMES:  # plain tensor attributes
            t = getattr(root, attr, None)
            if t is not None and hasattr(t, "dim") and t.dim() == 1:
                return t
    return None


class _WorkerSession:
    """Lives in the vLLM worker: hooks + recorder."""

    def __init__(self, model, spec: RoutingSpec, capacity: int, device: str):
        import torch

        from . import _native

        self.model, self.spec = model, spec
        self.layers = find_moe_layers(model)
        if not self.layers:
            raise UnsupportedRouting("no vLLM MoE layer (forward(hidden_states, router_logits)) found")
        if max(self.layers) > 255:
            raise UnsupportedRouting("layer ids past 255 do not fit the log")
        p = next(model.parameters())
        if device == "auto":
            device = f"cuda:{p.device.index or 0}" if p.is_cuda else "cpu"
        self.recorder = _native.Recorder(
            capacity=capacity, mode=spec.mode, device=device, normalize=spec.normalize,
            scale=spec.scale, n_group=spec.n_group, topk_group=spec.topk_group,
        )
        self.gpu = self.recorder.backend == "cuda"
        self.device = torch.device(device) if self.gpu else None
        if self.gpu:
            self.recorder.set_device_token_ids(True)
        self.offset = 0
        self.pass_tokens = 0
        self.handles: List[Any] = []
        parents = {id(m): p for p in model.modules() for m in p.children()}
        if spec.mode == "sigmoid":
            for i, mod in self.layers.items():
                b = _find_bias(mod, parents.get(id(mod)))
                if b is not None:
                    half = b.dtype in (torch.float16, torch.bfloat16)
                    self.recorder.set_bias(i, b.detach().float().cpu().numpy(), half=half)
        for i, mod in self.layers.items():
            self.handles.append(mod.register_forward_pre_hook(self._hook(i), with_kwargs=True))
        self.handles.append(model.register_forward_hook(self._end_of_pass))

    def _hook(self, layer_id: int):
        from . import route_and_log

        e = self.spec.num_experts

        def pre(_mod, args, kwargs):
            logits = kwargs.get("router_logits")
            if logits is None and len(args) > 1:
                logits = args[1]
            if logits is None or logits.shape[-1] != e:
                return
            logits = logits.reshape(-1, e)
            if self.gpu and logits.device != self.device:
                logits = logits.to(self.device, non_blocking=True)
            offset = 0 if self.gpu else self.offset
            route_and_log(logits, k=self.spec.k, layer_id=layer_id, token_offset=offset, recorder=self.recorder)
            self.pass_tokens = logits.shape[0]

        return pre

    def _end_of_pass(self, *_):
        n, self.pass_tokens = self.pass_tokens, 0
        if not n:
            return
        if self.gpu:
            import torch

            with torch.cuda.device(self.device):
                self.recorder.advance_tokens(n, stream=torch.cuda.current_stream().cuda_stream)
        self.offset += n

    def drain(self):
        from . import drain_with_stats

        return drain_with_stats(self.recorder)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _is_logging_rank() -> bool:
    """Tensor-parallel ranks all see the same (replicated) router logits:
    log on rank 0 only."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:  # not initialised / older vLLM: single worker
        return True


def _worker_attach(key, spec, capacity, device, model):
    if not _is_logging_rank():
        return None
    old = _WORKER.pop(key, None)
    if old is not None:
        old.detach()
    s = _WORKER[key] = _WorkerSession(model, spec, capacity, device)
    return sorted(s.layers)


def _worker_drain(key, model):
    s = _WORKER.get(key)
    return None if s is None else s.drain()


def _worker_detach(key, model):
    s = _WORKER.pop(key, None)
    if s is not None:
        s.detach()


class VllmSession:
    """Driver-side handle; the hooks and the log live in the vLLM worker."""

    def __init__(self, llm, key: str, spec: RoutingSpec, layers: List[int]):
        self.llm, self.key, self.spec, self.layers = llm, key, spec, layers

    def _call(self, fn):
        from functools import partial

        return [r for r in self.llm.apply_model(partial(fn, self.key)) if r is not None]

    def drain_with_stats(self, sort: bool = False):
        """``(records, dropped)``; resets the worker's log."""
        from . import RECORD_DTYPE, sort_records

        out = self._call(_worker_drain)
        if not out:
            return np.zeros(0, RECORD_DTYPE), 0
        log, dropped = out[0]
        return (sort_records(log) if sort else log), dropped

    def drain(self, sort: bool = False) -> np.ndarray:
        log, dropped = self.drain_with_stats(sort=sort)
        if dropped:
            import warnings

            warnings.warn(f"moescope: {dropped} routing records dropped (capacity full)", RuntimeWarning, stacklevel=2)
        return log

    def detach(self) -> None:
        self._call(_worker_detach)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()


def attach(llm, *, capacity: int = 1 << 24, device: str = "auto", spec: Optional[RoutingSpec] = None) -> VllmSession:
    """Logs every routing decision of a ``vllm.LLM`` (built with
    ``enforce_eager=True``). Returns a :class:`VllmSession`."""
    mc = llm.llm_engine.model_config
    if not getattr(mc, "enforce_eager", True):
        raise UnsupportedRouting("moescope.vllm needs LLM(..., enforce_eager=True): hooks cannot run in compiled graphs")
    spec = spec or routing_spec(mc.hf_config)
    if spec.kept_index is not None:
        raise UnsupportedRouting("capacity-factor models are not served by vLLM")
    from functools import partial

    key = f"s{id(llm)}"
    layers = [r for r in llm.apply_model(partial(_worker_attach, key, spec, capacity, device)) if r is not None]
    if not layers:
        raise UnsupportedRouting("no worker accepted the hooks")
    return VllmSession(llm, key, spec, layers[0])
