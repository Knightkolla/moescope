"""Routing health during training: load imbalance, dead experts, routing
drift and capacity drops, every N steps, from the moescope log.

    session = moescope.attach(model)
    monitor = moescope.monitor.RoutingMonitor(session, every=100, log_fn=wandb.log)
    for step, batch in enumerate(loader):
        loss = model(**batch).loss; loss.backward(); opt.step()
        monitor.step(step)

or with the Hugging Face Trainer::

    trainer = Trainer(..., callbacks=[moescope.monitor.MoescopeCallback(every=100)])

The log is drained into per-layer counts every step (one host copy), so
logging costs one kernel launch per MoE layer plus one copy per step. Recomputed forwards
(gradient checkpointing) are not logged twice.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


def _js_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def layer_counts(log: np.ndarray, num_experts: int) -> Dict[int, np.ndarray]:
    """``{layer: [kept decisions per expert..., capacity-dropped decisions]}``."""
    from . import CAPACITY_DROP

    out: Dict[int, np.ndarray] = {}
    if len(log) == 0:
        return out
    dropped = (log["flags"] & CAPACITY_DROP) != 0
    e = np.where(dropped, num_experts, log["expert_id"].astype(np.int64))  # drops go to the extra bin
    flat = log["layer_id"].astype(np.int64) * (num_experts + 1) + e
    c = np.bincount(flat, minlength=256 * (num_experts + 1)).reshape(256, num_experts + 1)
    for layer in np.flatnonzero(c.sum(1)).tolist():
        out[layer] = c[layer].astype(np.float64)
    return out


def window_stats(counts: Dict[int, np.ndarray], num_experts: int, previous: Optional[Dict[int, np.ndarray]] = None,
                 prefix: str = "moescope") -> tuple:
    """``(stats, loads)`` for one window, from :func:`layer_counts` output.

    Per layer ``L``: ``{prefix}/layer{L}/imbalance`` (busiest expert's load /
    fair share), ``/dead_experts`` (experts with no decision), ``/entropy``
    (load entropy / log2 E, 1 = perfectly even), ``/capacity_drop_rate`` and,
    with ``previous`` loads, ``/drift_js`` (JS divergence in bits from the
    previous window). Plus the mean (and max imbalance) over layers.
    """
    stats: Dict[str, float] = {}
    loads: Dict[int, np.ndarray] = {}
    per: Dict[str, List[float]] = {k: [] for k in ("imbalance", "dead_experts", "entropy", "capacity_drop_rate", "drift_js")}
    total = 0.0
    for layer, row in sorted(counts.items()):
        c, drops = row[:num_experts], row[num_experts]
        total += row.sum()
        p = c / max(c.sum(), 1)
        loads[layer] = p
        nz = p[p > 0]
        vals = {
            "imbalance": float(p.max() * num_experts),
            "dead_experts": float((c == 0).sum()),
            "entropy": float(-(nz * np.log2(nz)).sum() / np.log2(num_experts)) if num_experts > 1 else 1.0,
            "capacity_drop_rate": float(drops / max(row.sum(), 1)),
        }
        if previous is not None and layer in previous:
            vals["drift_js"] = _js_bits(p, previous[layer])
        for k, v in vals.items():
            stats[f"{prefix}/layer{layer}/{k}"] = v
            per[k].append(v)
    for k, vs in per.items():
        if vs:
            stats[f"{prefix}/mean/{k}"] = float(np.mean(vs))
    if per["imbalance"]:
        stats[f"{prefix}/max/imbalance"] = float(np.max(per["imbalance"]))
    stats[f"{prefix}/decisions"] = total
    return stats, loads


class RoutingMonitor:
    """Turns a :class:`moescope.hf.Session`'s log into scalar metrics every
    ``every`` steps (see :func:`window_stats`).

    The log is drained into per-layer counts on every step, so the GPU
    buffer only has to hold one step's decisions however long the window.

    log_fn: called as ``log_fn(stats, step=step)`` (``wandb.log`` fits;
        for TensorBoard pass ``lambda s, step: [w.add_scalar(k, v, step) for k, v in s.items()]``).
    refresh_bias: re-read DeepSeek-V3-style selection biases each window, since
        training updates them.
    """

    def __init__(self, session, every: int = 100, log_fn: Optional[Callable[..., Any]] = None,
                 prefix: str = "moescope", refresh_bias: bool = True):
        self.session, self.every, self.log_fn, self.prefix = session, max(1, every), log_fn, prefix
        self.refresh_bias = refresh_bias
        self.history: List[Dict[str, float]] = []
        self._previous: Optional[Dict[int, np.ndarray]] = None
        self._counts: Dict[int, np.ndarray] = {}
        self._lost = 0
        self._count = 0

    def _collect(self) -> None:
        log, lost = self.session.drain_with_stats()
        self._lost += lost
        for layer, row in layer_counts(log, self.session.spec.num_experts).items():
            self._counts[layer] = self._counts[layer] + row if layer in self._counts else row

    def step(self, step: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Call once per optimizer step. Returns the window's stats on the
        steps that close a window, else None."""
        self._count += 1
        self._collect()
        if self._count % self.every:
            return None
        return self.flush(step, collect=False)

    def flush(self, step: Optional[int] = None, collect: bool = True) -> Dict[str, float]:
        """Closes the current window now."""
        if collect:
            self._collect()
        stats, loads = window_stats(self._counts, self.session.spec.num_experts, self._previous, self.prefix)
        stats[f"{self.prefix}/records_lost"] = float(self._lost)
        self._counts, self._lost = {}, 0
        if loads:
            self._previous = loads
        if self.refresh_bias:
            self.session.refresh_bias()
        stats["step"] = float(self._count if step is None else step)
        self.history.append(stats)
        if self.log_fn is not None:
            self.log_fn({k: v for k, v in stats.items() if k != "step"}, step=int(stats["step"]))
        return stats


def _default_log_fn():
    try:
        import wandb

        if wandb.run is not None:
            return wandb.log
    except ImportError:
        pass
    return None


try:
    from transformers import TrainerCallback as _Base
except ImportError:  # transformers is optional
    _Base = object


class MoescopeCallback(_Base):
    """Hugging Face ``Trainer`` callback: attaches moescope at train start,
    logs routing health every ``every`` steps (to wandb if a run is active,
    or ``log_fn``), detaches at the end. Stats are kept in ``.history``."""

    def __init__(self, every: int = 100, log_fn: Optional[Callable[..., Any]] = None, capacity: int = 1 << 24,
                 device: str = "auto"):
        self.every, self.log_fn, self.capacity, self.device = every, log_fn, capacity, device
        self.session = self.monitor = None
        self.history: List[Dict[str, float]] = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from .hf import attach

        self.session = attach(model, capacity=self.capacity, device=self.device)
        self.monitor = RoutingMonitor(self.session, self.every, self.log_fn or _default_log_fn())
        self.history = self.monitor.history

    def on_step_end(self, args, state, control, **kwargs):
        if self.monitor is not None:
            self.monitor.step(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.session is not None:
            self.session.detach()
