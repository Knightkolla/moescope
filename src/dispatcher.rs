//! Token dispatch: group routed tokens by expert.
//!
//! After routing, an MoE layer "permutes" tokens so each expert can run one
//! dense GEMM over a contiguous block of its tokens. This module computes
//! that grouping on the CPU, plus the per-expert load histogram (how many
//! tokens each expert received), which is the main signal for spotting
//! load imbalance and dead experts.
//!
//! The GPU kernel produces the same [`DispatchPlan`], but fills each expert's
//! block in atomic order, so only the *set* of assignments per expert is
//! comparable, not their order.

use crate::router::Route;

/// Per-expert grouping of one routed batch.
///
/// An *assignment* is the flat index `a = token * k + j` into the routes
/// array, so a consumer can recover both the token (`a / k`) and the gate
/// (`routes[a].gate`) for every slot.
#[derive(Debug, Clone, PartialEq)]
pub struct DispatchPlan {
    num_experts: usize,
    /// Slots reserved per expert. Equals `num_tokens`: a token picks each
    /// expert at most once, so no expert can ever overflow its block.
    slots_per_expert: usize,
    /// Expert load histogram: `expert_counts[e]` tokens were sent to expert e.
    expert_counts: Vec<u32>,
    /// `[num_experts * slots_per_expert]`; expert e's assignments live in
    /// `gathered[e * slots_per_expert ..][..expert_counts[e]]`.
    gathered: Vec<u32>,
}

impl DispatchPlan {
    /// Assembles a plan from raw buffers (used to bring a GPU result back to
    /// host). Panics if the buffer sizes are inconsistent.
    pub fn from_parts(
        num_experts: usize,
        slots_per_expert: usize,
        expert_counts: Vec<u32>,
        gathered: Vec<u32>,
    ) -> Self {
        assert_eq!(expert_counts.len(), num_experts);
        assert_eq!(gathered.len(), num_experts * slots_per_expert);
        assert!(
            expert_counts
                .iter()
                .all(|&c| c as usize <= slots_per_expert)
        );
        DispatchPlan {
            num_experts,
            slots_per_expert,
            expert_counts,
            gathered,
        }
    }

    /// Number of experts in the layer.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Expert load histogram (tokens per expert).
    pub fn expert_load(&self) -> &[u32] {
        &self.expert_counts
    }

    /// Assignment indices (`token * k + j`) routed to `expert`, i.e. the
    /// rows this expert's FFN must process.
    pub fn assignments(&self, expert: usize) -> &[u32] {
        let start = expert * self.slots_per_expert;
        &self.gathered[start..start + self.expert_counts[expert] as usize]
    }
}

/// Groups a routed batch (`[token * k + j]` layout, as returned by
/// [`crate::router::route_batch`]) by expert, in token order.
///
/// Panics if `routes.len()` is not a multiple of `k` or an expert id is out
/// of range; both indicate routes that did not come from the router.
pub fn dispatch(routes: &[Route], k: usize, num_experts: usize) -> DispatchPlan {
    assert!(
        k > 0 && routes.len().is_multiple_of(k),
        "routes.len() must be a multiple of k"
    );
    let num_tokens = routes.len() / k;
    let mut expert_counts = vec![0u32; num_experts];
    let mut gathered = vec![0u32; num_experts * num_tokens];
    for (a, route) in routes.iter().enumerate() {
        let e = route.expert_id as usize;
        assert!(
            e < num_experts,
            "expert_id {e} out of range for {num_experts} experts"
        );
        let slot = expert_counts[e] as usize;
        gathered[e * num_tokens + slot] = a as u32;
        expert_counts[e] += 1;
    }
    DispatchPlan {
        num_experts,
        slots_per_expert: num_tokens,
        expert_counts,
        gathered,
    }
}
