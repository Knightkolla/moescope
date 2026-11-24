"""One-line hookup for Hugging Face MoE models.

    import moescope
    with moescope.attach(model) as session:   # finds every MoE router
        model.generate(**inputs)
    log = session.drain()                     # token_id, layer_id, expert_id, gate

`attach` reads the routing rule from the model's config (softmax / sigmoid,
normalization, scaling, group-limited routing), registers per-layer
selection biases, and hooks each router. Each hook makes one kernel launch
on the router's own logits, so routing is never recomputed in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import _native

# Where decoder layers keep their router, first match wins.
ROUTER_PATHS = (
    "mlp.gate",  # OLMoE, Qwen-MoE, DeepSeek, GLM-4.5, MiniMax, dots1
    "block_sparse_moe.router",  # Granite
    "block_sparse_moe.gate",  # Mixtral
    "mlp.router",
    "feed_forward.router",  # Llama 4
    "feed_forward.gate",  # LFM2-MoE
)


class UnsupportedRouting(NotImplementedError):
    """The model routes in a way moescope cannot reproduce exactly."""


@dataclass(frozen=True)
class RoutingSpec:
    """How one model family turns router logits into experts and gates."""

    num_experts: int
    k: int
    mode: str  # "softmax_topk" | "topk_softmax" | "sigmoid"
    normalize: bool = False
    scale: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    logits_index: Optional[int] = None  # position of the logits in the router's output tuple


def _text_config(config):
    return getattr(config, "text_config", None) or config


def routing_spec(config) -> RoutingSpec:
    """Derives the routing rule from a Hugging Face config.

    Raises :class:`UnsupportedRouting` for rules moescope does not implement
    (DeepSeek-V2 ``group_limited_greedy``, ERNIE-4.5's biased softmax), so
    a log is never silently wrong.
    """
    c = _text_config(config)
    t = c.model_type
    e = next(
        (getattr(c, a) for a in ("n_routed_experts", "num_local_experts", "num_experts") if getattr(c, a, None)),
        None,
    )
    k = getattr(c, "num_experts_per_tok", None)
    if not e or not k:
        raise UnsupportedRouting(f"{t}: config has no expert count / num_experts_per_tok; is this an MoE model?")
    scale = float(getattr(c, "routed_scaling_factor", 1.0) or 1.0)
    norm = bool(getattr(c, "norm_topk_prob", False))

    if t.startswith("ernie4_5"):
        raise UnsupportedRouting("ERNIE-4.5 selects on softmax + a learned bias (moe_statics); not supported yet")
    if t == "deepseek_v2":
        if getattr(c, "topk_method", "greedy") != "greedy":
            raise UnsupportedRouting("DeepSeek-V2 'group_limited_greedy' routing is not supported yet")
        return RoutingSpec(e, k, "softmax_topk", scale=scale)
    if t.startswith("llama4"):
        # topk on logits, then sigmoid of the selected logits; the router
        # returns (scores, logits).
        return RoutingSpec(e, k, "sigmoid", logits_index=1)
    sigmoid = getattr(c, "scoring_func", None) == "sigmoid" or t in ("deepseek_v3", "glm4_moe", "dots1", "kimi_k2", "lfm2_moe")
    if sigmoid:
        n_group = int(getattr(c, "n_group", 1) or 1)
        topk_group = int(getattr(c, "topk_group", 1) or 1)
        return RoutingSpec(e, k, "sigmoid", normalize=norm, scale=scale, n_group=n_group, topk_group=topk_group)
    if t in ("mixtral", "granitemoe", "granitemoeshared", "granitemoehybrid", "minimax", "hunyuan_v1_moe"):
        # top-k on logits, softmax over the k (or the equivalent normalized softmax probs)
        return RoutingSpec(e, k, "topk_softmax")
    # OLMoE, Qwen2/3-MoE and friends: softmax over all experts, then top-k,
    # renormalized iff norm_topk_prob (which equals a softmax over the k).
    return RoutingSpec(e, k, "topk_softmax" if norm else "softmax_topk")


def _decoder_layers(model):
    """The model's list of decoder layers (``*.layers`` ModuleList)."""
    import torch

    best = None
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "layers" and isinstance(mod, torch.nn.ModuleList):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise UnsupportedRouting("could not find the model's decoder layers")
    return best


def _walk(root, path):
    mod = root
    for part in path.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    return mod


def selection_bias(layer, router_path: str):
    """The per-expert selection bias of one MoE layer, or None: DeepSeek-V3 /
    GLM keep it on the router (``e_score_correction_bias``), LFM2-MoE on the
    MoE block (``expert_bias``, used when ``use_expert_bias``)."""
    router = _walk(layer, router_path)
    bias = getattr(router, "e_score_correction_bias", None)
    if bias is not None:
        return bias
    block = _walk(layer, router_path.rsplit(".", 1)[0]) if "." in router_path else None
    if block is not None and getattr(block, "use_expert_bias", True):
        return getattr(block, "expert_bias", None)
    return None


def find_routers(model, num_experts: int) -> Dict[int, Any]:
    """``{decoder_layer_index: router_module}`` for every MoE layer.

    Dense layers (e.g. DeepSeek's first layers) are skipped, so layer ids in
    the log are the model's real layer indices.
    """
    routers = {}
    for i, layer in enumerate(_decoder_layers(model)):
        for path in ROUTER_PATHS:
            mod = _walk(layer, path)
            w = getattr(mod, "weight", None) if mod is not None else None
            if w is None and mod is not None:
                w = getattr(getattr(mod, "layer", None), "weight", None)
            if w is not None and w.shape[0] == num_experts:
                routers[i] = mod
                mod._moescope_path = path  # where selection_bias() looks
                mod._moescope_layer = layer
                break
    if not routers:
        raise UnsupportedRouting("no MoE router found; add its path to moescope.hf.ROUTER_PATHS")
    return routers


def _pick(out, spec: RoutingSpec):
    """(logits, indices, weights) from a router's output, by dtype and shape."""
    import torch

    outs = out if isinstance(out, tuple) else (out,)
    tensors = [t for t in outs if isinstance(t, torch.Tensor)]
    if spec.logits_index is not None:
        logits = outs[spec.logits_index]
    else:
        logits = next((t for t in tensors if t.is_floating_point() and t.shape[-1] == spec.num_experts), None)
    indices = next((t for t in tensors if not t.is_floating_point() and t.shape[-1] == spec.k), None)
    weights = next((t for t in tensors if t.is_floating_point() and t.shape[-1] == spec.k), None)
    return logits, indices, weights


class Session:
    """Hooks on one model's routers, plus the recorder they log into.

    ``token_id`` in the log counts tokens in the order the model processed
    them (all of a forward pass's tokens, then the next pass's), so it also
    works across ``generate`` steps.
    """

    def __init__(self, model, recorder, routers: Dict[int, Any], spec: RoutingSpec, check: bool):
        self.model, self.recorder, self.routers, self.spec, self.check = model, recorder, routers, spec, check
        self.offset = 0
        self._pass_tokens = 0
        self._handles: List[Any] = []
        self._reference: List[tuple] = []
        self._device = None
        if recorder.backend == "cuda":
            import torch

            self._device = torch.device("cuda", _router_weight(next(iter(routers.values()))).device.index or 0)

    # -- hooks -------------------------------------------------------------
    def _router_hook(self, layer_id: int):
        from . import route_and_log

        def hook(_mod, _inputs, out):
            logits, indices, weights = _pick(out, self.spec)
            if logits is None:
                raise UnsupportedRouting(f"layer {layer_id}: router output has no [tokens, {self.spec.num_experts}] logits")
            logits = logits.reshape(-1, self.spec.num_experts)
            if self._device is not None and logits.device != self._device:
                logits = logits.to(self._device)
                import torch

                torch.cuda.current_stream(self._device).synchronize()
            route_and_log(logits, k=self.spec.k, layer_id=layer_id, token_offset=self.offset, recorder=self.recorder)
            self._pass_tokens = logits.shape[0]
            if self.check and indices is not None:
                w = None if weights is None else weights.reshape(-1, self.spec.k).float().cpu()
                self._reference.append((layer_id, self.offset, indices.reshape(-1, self.spec.k).cpu(), w))

        return hook

    def _model_hook(self, *_args):
        self.offset += self._pass_tokens
        self._pass_tokens = 0

    def _attach(self):
        for layer_id, router in self.routers.items():
            self._handles.append(router.register_forward_hook(self._router_hook(layer_id)))
        self._handles.append(self.model.register_forward_hook(self._model_hook))
        return self

    # -- public API ----------------------------------------------------------
    def refresh_bias(self) -> None:
        """Re-reads every router's selection bias (DeepSeek-V3's
        ``e_score_correction_bias``). Called by :func:`attach`; call it again
        if training updates the bias."""
        if self.spec.mode != "sigmoid":
            return
        for layer_id, router in self.routers.items():
            bias = selection_bias(router._moescope_layer, router._moescope_path)
            if bias is not None:
                self.recorder.set_bias(layer_id, bias.detach().float().cpu().numpy())

    def detach(self) -> None:
        """Removes every hook. The log stays readable with :meth:`drain`."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    def drain_with_stats(self):
        """``(records, dropped)``, like :func:`moescope.drain_with_stats`."""
        from . import drain_with_stats

        return drain_with_stats(self.recorder)

    def drain(self) -> np.ndarray:
        """All records logged so far (and resets the log)."""
        from . import drain

        return drain(self.recorder)

    def parity(self, log: np.ndarray) -> Optional[Dict[str, float]]:
        """Compares ``log`` with what the model's routers chose (needs
        ``attach(..., check=True)`` and routers that return their indices).

        Returns ``{"match_rate", "checked", "max_gate_err"}``: the share of
        (layer, token) expert *sets* that are identical, how many were
        compared, and the largest gate difference over matching sets.
        """
        if not self._reference:
            return None
        k = self.spec.k
        srt = log[np.lexsort((log["expert_id"], log["token_id"], log["layer_id"]))]
        ids = srt["expert_id"].reshape(-1, k).astype(np.int64)
        gates = srt["gate"].reshape(-1, k)
        row = {(int(l), int(t)): i for i, (l, t) in enumerate(zip(srt["layer_id"][::k], srt["token_id"][::k]))}
        match = total = 0
        max_err = 0.0
        for layer, offset, idx, w in self._reference:
            idx = idx.numpy().astype(np.int64)
            order = np.argsort(idx, axis=1)
            ref_ids = np.take_along_axis(idx, order, 1)
            for i in range(len(idx)):
                r = row.get((layer, offset + i))
                total += 1
                if r is None or not np.array_equal(ids[r], ref_ids[i]):
                    continue
                match += 1
                if w is not None:
                    ref_w = np.take_along_axis(w[i : i + 1].numpy(), order[i : i + 1], 1)[0]
                    max_err = max(max_err, float(np.abs(gates[r] - ref_w).max()))
        return {"match_rate": match / max(total, 1), "checked": total, "max_gate_err": max_err}


def _router_weight(router):
    w = getattr(router, "weight", None)
    return w if w is not None else router.layer.weight  # transformers 4.x Granite gating


def attach(
    model,
    *,
    capacity: int = 1 << 24,
    device: str = "auto",
    check: bool = False,
    spec: Optional[RoutingSpec] = None,
) -> Session:
    """Hooks every MoE router of a Hugging Face model and logs its routing.

    capacity: records held between drains (12 bytes each).
    device: "auto" (the GPU the routers live on, else CPU), "cpu", "cuda:N".
    check: also keep the model's own expert choices so
        :meth:`Session.parity` can verify the log (costs a host copy per layer).
    spec: override the routing rule read from ``model.config``.
    """
    spec = spec or routing_spec(model.config)
    routers = find_routers(model, spec.num_experts)
    if device == "auto":
        w = _router_weight(next(iter(routers.values())))
        device = f"cuda:{w.device.index or 0}" if w.is_cuda else "cpu"
    recorder = _native.Recorder(
        capacity=capacity,
        mode=spec.mode,
        device=device,
        normalize=spec.normalize,
        scale=spec.scale,
        n_group=spec.n_group,
        topk_group=spec.topk_group,
    )
    session = Session(model, recorder, routers, spec, check)
    session.refresh_bias()
    return session._attach()
