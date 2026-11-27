"""Turning a routing log into decisions: which experts a workload needs,
how many can stay on the GPU, which could be pruned.

All functions take a ``usage`` array ``[layers, experts]`` (decision counts
or gate mass, e.g. from :func:`expert_usage`) and work per layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def expert_usage(
    log: np.ndarray,
    num_experts: int,
    weight: str = "count",
    layers: Optional[Sequence[int]] = None,
    include_capacity_drops: bool = False,
) -> Tuple[List[int], np.ndarray]:
    """``(layer_ids, usage[len(layer_ids), num_experts])`` from a log.

    weight: "count" (decisions) or "gate" (sum of gate weights, i.e. how much
        each expert contributes to the layer's output).
    include_capacity_drops: count decisions flagged ``CAPACITY_DROP`` (the
        expert never computed them). Off by default.
    """
    from . import CAPACITY_DROP

    if not include_capacity_drops:
        log = log[(log["flags"] & CAPACITY_DROP) == 0]
    layer_ids = sorted(set(log["layer_id"].tolist())) if layers is None else list(layers)
    row = np.full(256, -1, np.int64)
    row[layer_ids] = np.arange(len(layer_ids))
    r = row[log["layer_id"]]
    keep = r >= 0
    flat = r[keep] * num_experts + log["expert_id"][keep]
    w = None if weight == "count" else log["gate"][keep].astype(np.float64)
    if weight not in ("count", "gate"):
        raise ValueError("weight must be 'count' or 'gate'")
    u = np.bincount(flat, weights=w, minlength=len(layer_ids) * num_experts)
    return layer_ids, u.reshape(len(layer_ids), num_experts).astype(np.float64)


def experts_for_coverage(usage: np.ndarray, coverage: float = 0.99) -> List[np.ndarray]:
    """Per layer, the smallest set of experts (busiest first) that carries at
    least ``coverage`` of the layer's usage."""
    out = []
    for u in usage:
        order = np.argsort(-u, kind="stable")
        share = np.cumsum(u[order]) / max(u.sum(), 1e-30)
        n = int(np.searchsorted(share, coverage - 1e-12) + 1)
        out.append(np.sort(order[: min(n, len(u))]))
    return out


def resident_hit_rate(usage: np.ndarray, budget: int, plan_usage: Optional[np.ndarray] = None) -> np.ndarray:
    """Per layer, the share of ``usage`` served if only ``budget`` experts
    per layer stay on the GPU, chosen as the busiest under ``plan_usage``
    (default: ``usage`` itself; pass another workload's usage to test a plan
    made on one domain against another)."""
    plan = usage if plan_usage is None else plan_usage
    hits = []
    for u, p in zip(usage, plan):
        keep = np.argsort(-p, kind="stable")[:budget]
        hits.append(u[keep].sum() / max(u.sum(), 1e-30))
    return np.array(hits)


def budget_curve(usage: np.ndarray, plan_usage: Optional[np.ndarray] = None) -> np.ndarray:
    """``[experts + 1]``: mean hit rate over layers for budgets 0..E."""
    e = usage.shape[1]
    return np.array([resident_hit_rate(usage, b, plan_usage).mean() for b in range(e + 1)])


def coverage_table(usage_by_domain: Dict[str, np.ndarray], coverages=(0.9, 0.99, 0.999)) -> Dict[str, Dict[float, float]]:
    """``{domain: {coverage: mean experts needed per layer}}``."""
    return {
        d: {c: float(np.mean([len(s) for s in experts_for_coverage(u, c)])) for c in coverages}
        for d, u in usage_by_domain.items()
    }
