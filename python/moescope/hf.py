"""One-line hookup for Hugging Face MoE models.

    import moescope
    with moescope.attach(model) as session:   # finds every MoE router
        model.generate(**inputs)
    log = session.drain()                     # token_id, layer_id, flags, expert_id, gate

`attach` reads the routing rule from the model's config (softmax / sigmoid,
normalization, scaling, group-limited routing), registers per-layer
selection biases, and hooks each router. Each hook makes one kernel launch
on the router's own logits, so routing is never recomputed in Python.

Capacity-factor models (Switch Transformer, NLLB-MoE) can drop a token
that the router sent to a full expert. For those the hook also hands the
kernel the router's dispatch mask, so such decisions are logged with
``flags & moescope.CAPACITY_DROP``: the log says what the router chose *and*
whether the expert computed it.
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
    "layer.1.mlp.router",  # Switch Transformer encoder blocks
    "layer.2.mlp.router",  # Switch Transformer decoder blocks (after cross-attention)
    "ffn.router",  # NLLB-MoE
)

# ModuleLists that hold a model's transformer layers ("block" for T5-style).
_STACK_NAMES = ("layers", "block")


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
    # Position of the dispatch mask after capacity (nonzero = the expert
    # computes the token) in the router's output, for capacity-factor models.
    kept_index: Optional[int] = None


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
    if t == "switch_transformers":
        # top-1 on softmax probs; tokens past an expert's capacity (per
        # sequence) are masked out. Router returns (dispatch, probs, logits).
        return RoutingSpec(c.num_experts, 1, "softmax_topk", logits_index=2, kept_index=0)
    if t in ("nllb_moe", "nllb-moe"):
        if c.second_expert_policy == "sampling":
            raise UnsupportedRouting("NLLB-MoE second_expert_policy='sampling' adds Gumbel noise to the logits in place")
        # top-2 by logits, gates renormalized over the two; capacity drops
        # zero the combine weights. Router returns (top1_mask, combine, logits).
        return RoutingSpec(c.num_experts, 2, "topk_softmax", logits_index=2, kept_index=1)
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


def _layer_stacks(model):
    """``[(name, owner_module, layers)]`` for every ``*.layers`` / ``*.block``
    ModuleList, in module order (an encoder-decoder has two)."""
    import torch

    mods = dict(model.named_modules())
    stacks = []
    for name, mod in mods.items():
        if name.split(".")[-1] in _STACK_NAMES and isinstance(mod, torch.nn.ModuleList):
            owner = mods[name.rsplit(".", 1)[0]] if "." in name else model
            stacks.append((name, owner, mod))
    if not stacks:
        raise UnsupportedRouting("could not find the model's decoder layers")
    return stacks


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
    """``{layer_index: router_module}`` for every MoE layer.

    Dense layers (e.g. DeepSeek's first layers) are skipped, so layer ids in
    the log are the model's real layer indices. With several layer stacks
    that hold routers (encoder-decoder), the decoder's ids continue after
    the encoder's: encoder layer i is i, decoder layer j is
    ``len(encoder) + j``.
    """
    routers: Dict[int, Any] = {}
    base = 0
    stacks = 0
    for _name, owner, layers in _layer_stacks(model):
        found = {}
        for i, layer in enumerate(layers):
            for path in ROUTER_PATHS:
                mod = _walk(layer, path)
                w = _router_weight(mod) if mod is not None else None
                if w is not None and w.shape[0] == num_experts:
                    found[base + i] = mod
                    mod._moescope_path = path  # where selection_bias() looks
                    mod._moescope_layer = layer
                    mod._moescope_stack = owner  # forward of a whole pass
                    break
        if found:
            routers.update(found)
            base += len(layers)
            stacks += 1
    if not routers:
        raise UnsupportedRouting("no MoE router found; add its path to moescope.hf.ROUTER_PATHS")
    if max(routers) > 255:
        raise UnsupportedRouting(f"layer id {max(routers)} does not fit the log's u8 layer_id")
    return routers


def _pick(out, spec: RoutingSpec):
    """(logits, indices, weights) from a router's output, by dtype and shape."""
    import torch

    outs = out if isinstance(out, tuple) else (out,)
    if spec.kept_index is not None:  # capacity models: no top-k tensors
        return outs[spec.logits_index], None, None
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
    works across ``generate`` steps. In an encoder-decoder, encoder and
    decoder passes share that counter, each pass getting fresh ids.
    """

    def __init__(self, model, recorder, routers: Dict[int, Any], spec: RoutingSpec, check: bool):
        self.model, self.recorder, self.routers, self.spec, self.check = model, recorder, routers, spec, check
        self.offset = 0
        self._pass_tokens: Dict[int, int] = {}  # id(stack owner) -> tokens in its current pass
        self._handles: List[Any] = []
        self._reference: List[tuple] = []
        self._reference_source = "router"
        self._device = None
        if recorder.backend == "cuda":
            import torch

            self._device = torch.device("cuda", _router_weight(next(iter(routers.values()))).device.index or 0)
            # Token ids come from a device counter moved by a tiny kernel after
            # each pass, so they stay right when the model runs from a
            # captured CUDA graph (vLLM / SGLang style decoding).
            recorder.set_device_token_ids(True)
        self.detach()

    def drain_with_stats(self):
        """``(records, dropped)``, like :func:`moescope.drain_with_stats`."""
        from . import drain_with_stats

        return drain_with_stats(self.recorder)

    def drain(self) -> np.ndarray:
        """All records logged so far (and resets the log)."""
        from . import drain

        return drain(self.recorder)

    def parity(self, log: np.ndarray) -> Optional[Dict[str, Any]]:
        """Checks ``log`` against what the model's *experts* computed (needs
        ``attach(..., check=True)``).

        The reference is taken from the experts' own inputs where the model
        exposes them (the ``top_k_index`` every transformers 5 MoE block
        hands its experts, or the rows each expert module of Switch /
        NLLB-MoE actually received), else from the router's returned indices.
        A (layer, token) matches when the experts that computed it are
        exactly the logged decisions without ``CAPACITY_DROP``, each once.

        Returns ``{"match_rate", "checked", "max_gate_err", "capacity_drops",
        "source"}``: the share of matching (layer, token) pairs, how many were
        compared, the largest gate difference over matching tokens with no
        capacity drop, the number of logged decisions flagged as dropped, and
        ``"experts"`` or ``"router"``.
        """
        from . import CAPACITY_DROP

        if not self._reference:
            return None
        e_count = self.spec.num_experts
        kept = (log["flags"] & CAPACITY_DROP) == 0
        rows: Dict[tuple, List[int]] = {}
        for i, key in enumerate(zip(log["layer_id"].tolist(), log["token_id"].tolist())):
            rows.setdefault(key, []).append(i)
        experts = log["expert_id"].astype(np.int64)
        match = total = 0
        max_err = 0.0
        for layer, offset, computed, weights in self._reference:
            for t in range(computed.shape[0]):
                total += 1
                idx = rows.get((layer, offset + t))
                if idx is None:
                    continue
                mine = np.zeros(e_count, np.int64)
                np.add.at(mine, experts[idx][kept[idx]], 1)
                if not np.array_equal(mine, computed[t]):
                    continue
                match += 1
                if weights is not None and kept[idx].all():
                    ref = weights[t, experts[idx]]
                    ok = ~np.isnan(ref)
                    if ok.any():
                        max_err = max(max_err, float(np.abs(log["gate"][idx][ok] - ref[ok]).max()))
        return {
            "match_rate": match / max(total, 1),
            "checked": total,
            "max_gate_err": max_err,
            "capacity_drops": int((~kept).sum()),
            "source": self._reference_source,
        }

    # -- hooks -------------------------------------------------------------
    def _router_hook(self, layer_id: int, stack: Any):
        from . import route_and_log

        spec = self.spec

        def hook(_mod, _inputs, out):
            logits, indices, weights = _pick(out, spec)
            if logits is None:
                raise UnsupportedRouting(f"layer {layer_id}: router output has no [tokens, {spec.num_experts}] logits")
            logits = logits.reshape(-1, spec.num_experts)
            kept = None
            if spec.kept_index is not None:
                kept = out[spec.kept_index].reshape(-1, spec.num_experts)
            if self._device is not None and logits.device != self._device:
                # Cross-GPU (device_map): torch orders the copy on both
                # devices' current streams, and the kernel launches on the
                # destination's current stream, so no host sync is needed.
                logits = logits.to(self._device, non_blocking=True)
                kept = None if kept is None else kept.to(self._device, non_blocking=True)
            offset = 0 if self._device is not None else self.offset  # GPU: device counter
            route_and_log(logits, k=spec.k, layer_id=layer_id, token_offset=offset, recorder=self.recorder, kept=kept)
            self._pass_tokens[id(stack)] = logits.shape[0]
            if self.check and not getattr(_mod, "_moescope_experts_checked", False) and indices is not None:
                n = logits.shape[0]
                computed = np.zeros((n, spec.num_experts), np.int64)
                ind = indices.reshape(n, -1).cpu().numpy().astype(np.int64)
                np.add.at(computed, (np.repeat(np.arange(n), ind.shape[1]), ind.ravel()), 1)
                w = None
                if weights is not None:
                    w = np.full((n, spec.num_experts), np.nan, np.float32)
                    w[np.arange(n)[:, None], ind] = weights.reshape(n, -1).float().cpu().numpy()
                self._reference.append((layer_id, self.offset, computed, w))

        return hook

    def _stack_hook(self, stack: Any):
        key = id(stack)

        def hook(*_args):
            n = self._pass_tokens.pop(key, 0)
            if not n:
                return
            if self._device is not None:
                import torch

                with torch.cuda.device(self._device):
                    self.recorder.advance_tokens(n, stream=torch.cuda.current_stream().cuda_stream)
            self.offset += n

        return hook

    def _experts_hooks(self, layer_id: int, router: Any) -> bool:
        """With ``check``, hooks the MoE block's experts so parity compares
        against what they computed. False if this block has no usable experts."""
        import torch

        block = _walk(router._moescope_layer, router._moescope_path.rsplit(".", 1)[0]) if "." in router._moescope_path else None
        experts = getattr(block, "experts", None)
        if experts is None:
            return False
        spec = self.spec
        children = _expert_children(experts, spec.num_experts)
        if children is not None:
            # One module per expert (Switch, NLLB-MoE): record the rows each
            # expert receives and map them back to tokens.
            state: Dict[str, Any] = {}

            def on_block(_mod, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                h = h.reshape(-1, h.shape[-1]).detach().float().cpu().numpy()
                where: Dict[bytes, List[int]] = {}
                for t in range(len(h)):
                    where.setdefault(h[t].tobytes(), []).append(t)
                rw = args[2] if len(args) > 2 else None
                computed = np.zeros((len(h), spec.num_experts), np.int64)
                w = None
                if rw is not None and rw.is_floating_point():
                    rw = rw.reshape(len(h), -1).detach().float().cpu().numpy()
                    if rw.shape[1] == spec.num_experts:
                        w = np.where(rw != 0, rw, np.nan).astype(np.float32)
                state.update(where=where, computed=computed, rw=rw, w=w)
                self._reference.append((layer_id, self.offset, computed, w))

            def on_expert(e):
                def pre(_mod, args):
                    x = args[0].reshape(-1, args[0].shape[-1]).detach().float().cpu().numpy()
                    computed = state["computed"]
                    for row in x:
                        ts = state["where"].get(row.tobytes(), ())
                        if ts:  # identical rows: credit the first not yet sent to e
                            t = next((t for t in ts if computed[t, e] == 0), ts[0])
                            computed[t, e] += 1

                return pre

            def on_block_done(_mod, _args, _out):
                rw, computed = state.get("rw"), state.get("computed")
                if state.get("w") is None and rw is not None and rw.shape[1] == 1 and computed is not None:
                    w = np.full(computed.shape, np.nan, np.float32)
                    t, e = np.nonzero(computed)
                    w[t, e] = rw[t, 0]
                    ref = self._reference[-1]
                    self._reference[-1] = (ref[0], ref[1], ref[2], w)

            self._handles.append(experts.register_forward_pre_hook(on_block, with_kwargs=True))
            self._handles.append(experts.register_forward_hook(on_block_done))
            for e, child in children:
                self._handles.append(child.register_forward_pre_hook(on_expert(e)))
            return True

        # Fused experts (transformers 5): forward(hidden_states, top_k_index, top_k_weights).
        def on_fused(_mod, args, kwargs):
            ind = args[1] if len(args) > 1 else kwargs.get("top_k_index")
            wts = args[2] if len(args) > 2 else kwargs.get("top_k_weights")
            if not isinstance(ind, torch.Tensor) or ind.is_floating_point() or ind.shape[-1] != spec.k:
                return
            n = ind.reshape(-1, spec.k).shape[0]
            ind = ind.reshape(n, spec.k).cpu().numpy().astype(np.int64)
            computed = np.zeros((n, spec.num_experts), np.int64)
            np.add.at(computed, (np.repeat(np.arange(n), spec.k), ind.ravel()), 1)
            w = None
            if isinstance(wts, torch.Tensor) and wts.shape[-1] == spec.k:
                w = np.full((n, spec.num_experts), np.nan, np.float32)
                w[np.arange(n)[:, None], ind] = wts.reshape(n, spec.k).detach().float().cpu().numpy()
            self._reference.append((layer_id, self.offset, computed, w))

        probe = getattr(experts, "forward", None)
        import inspect

        try:
            params = list(inspect.signature(probe).parameters)
        except (TypeError, ValueError):
            return False
        if len(params) < 2:
            return False
        self._handles.append(experts.register_forward_pre_hook(on_fused, with_kwargs=True))
        return True

    def _attach(self):
        stacks = {}
        experts_checked = bool(self.check)
        for layer_id, router in self.routers.items():
            stack = router._moescope_stack
            stacks[id(stack)] = stack
            self._handles.append(router.register_forward_hook(self._router_hook(layer_id, stack)))
            ok = self.check and self._experts_hooks(layer_id, router)
            router._moescope_experts_checked = ok
            experts_checked = experts_checked and ok
        if self.check:
            self._reference_source = "experts" if experts_checked else "router"
        for stack in stacks.values():
            self._handles.append(stack.register_forward_hook(self._stack_hook(stack)))
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
                import torch

                half = bias.dtype in (torch.float16, torch.bfloat16)
                self.recorder.set_bias(layer_id, bias.detach().float().cpu().numpy(), half=half)

    def detach(self) -> None:
        """Removes every hook. The log stays readable with :meth:`drain`."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()


def _expert_children(experts, num_experts: int):
    """``[(expert_index, module)]`` if ``experts`` holds one module per
    expert (``expert_{i}`` ModuleDict or a ModuleList), else None."""
    import torch

    if isinstance(experts, torch.nn.ModuleDict):
        kids = [(int(name.rsplit("_", 1)[-1]), m) for name, m in experts.items() if name.rsplit("_", 1)[-1].isdigit()]
    elif isinstance(experts, torch.nn.ModuleList):
        kids = list(enumerate(experts))
    else:
        return None
    return kids if len(kids) == num_experts else None


def _router_weight(router):
    """The router's ``[num_experts, hidden]`` weight, or None."""
    for path in ("weight", "layer.weight", "classifier.weight"):  # 4.x Granite: .layer; Switch / NLLB: .classifier
        w = _walk(router, path)
        if w is not None and hasattr(w, "shape"):
            return w
    return None


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
    check: also record what the experts actually computed so
        :meth:`Session.parity` can verify the log (costs host copies per layer;
        for testing, not serving).
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
