//! CPU reference router for a Mixture-of-Experts (MoE) layer.
//!
//! In an MoE layer a small linear "gate" produces one logit per expert for
//! every token. The router turns those logits into a routing decision: which
//! `k` experts process the token, and with what weight (the *gate*) each
//! expert's output is mixed back into the residual stream.
//!
//! This module is the ground truth the GPU kernels are tested against, so the
//! ordering rules here (ties, NaN) are deliberately simple and mirrored
//! exactly in `gpu/kernels.rs`.
//!
//! Selection is O(E log E) per token via a full sort. E (experts per layer)
//! is small in practice (8-128), so partial selection
//! (`select_nth_unstable_by`) is not worth the complexity yet.

use std::cmp::Ordering;
use std::fmt;

/// One routing decision for one token: send it to `expert_id`, and scale that
/// expert's output by `gate` when combining.
///
/// `#[repr(C)]` so the GPU kernel can write it directly (it is `RouteEntry`
/// in the CUDA source: `{ unsigned int expert_id; float gate; }`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Route {
    pub expert_id: u32,
    pub gate: f32,
}

/// How gate weights are derived from router logits.
///
/// Both modes select the same experts (softmax is monotonic, and we always
/// select on raw logits); they differ only in the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateMode {
    /// Softmax over *all* experts, then keep the top-k probabilities as-is.
    /// Gates sum to <= 1. Used by Switch Transformer / OLMoE
    /// (`norm_topk_prob = False`).
    #[default]
    SoftmaxThenTopK,
    /// Pick the top-k logits, then softmax over just those k, so gates are
    /// renormalized and sum to 1. Used by Mixtral.
    TopKThenSoftmax,
}

/// Reasons a routing request is rejected instead of silently producing
/// garbage gates.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteError {
    /// `k` must be at least 1: a token routed to zero experts is dropped.
    ZeroK,
    /// Cannot pick more distinct experts than the layer has.
    KExceedsExperts { k: usize, num_experts: usize },
    /// A layer needs at least one expert.
    NoExperts,
    /// The flat logits buffer is not a whole number of `num_experts` rows.
    RaggedLogits { len: usize, num_experts: usize },
    /// A NaN or infinite logit; softmax over it would be meaningless.
    NonFiniteLogit { token: usize, expert: usize, value: f32 },
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouteError::ZeroK => write!(f, "k must be >= 1"),
            RouteError::KExceedsExperts { k, num_experts } => {
                write!(f, "k = {k} exceeds num_experts = {num_experts}")
            }
            RouteError::NoExperts => write!(f, "num_experts must be >= 1"),
            RouteError::RaggedLogits { len, num_experts } => write!(
                f,
                "logits length {len} is not a multiple of num_experts = {num_experts}"
            ),
            RouteError::NonFiniteLogit { token, expert, value } => write!(
                f,
                "non-finite logit {value} at token {token}, expert {expert}"
            ),
        }
    }
}

impl std::error::Error for RouteError {}

/// Total, deterministic ranking of `(expert_index, score)` pairs: higher score
/// first, and on an exact tie the *lower expert index* wins. NaN ranks below
/// every number, so it never panics and is only picked when nothing else is
/// left. The GPU kernel implements the same rule with a strict `>` scan over
/// experts in index order.
fn rank_order(a: (usize, f32), b: (usize, f32)) -> Ordering {
    let key = |s: f32| if s.is_nan() { f32::NEG_INFINITY } else { s };
    let (sa, sb) = (key(a.1), key(b.1));
    if sa > sb {
        Ordering::Less
    } else if sa < sb {
        Ordering::Greater
    } else {
        a.0.cmp(&b.0)
    }
}

/// Indices of the `k` highest scores, best first (ties -> lower index).
///
/// This is the expert *selection* step of an MoE router. If `k` exceeds the
/// number of scores, all indices are returned. Never panics, even on NaN.
pub fn top_k_route(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|&a, &b| rank_order(a, b));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// [`top_k_route`] applied to each token of a batch (one `Vec` per token).
pub fn batch_top_k_route(scores: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    scores
        .iter()
        .map(|token_scores| top_k_route(token_scores, k))
        .collect()
}

/// Numerically stable softmax: turns router logits into a probability
/// distribution over experts. Subtracting the max keeps `exp` from
/// overflowing for large logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Validates the shape parameters shared by every routing entry point.
pub fn check_shape(num_experts: usize, k: usize) -> Result<(), RouteError> {
    if num_experts == 0 {
        return Err(RouteError::NoExperts);
    }
    if k == 0 {
        return Err(RouteError::ZeroK);
    }
    if k > num_experts {
        return Err(RouteError::KExceedsExperts { k, num_experts });
    }
    Ok(())
}

/// Rejects NaN / +-inf logits, reporting where the first one is.
pub fn check_finite(logits: &[f32], token: usize) -> Result<(), RouteError> {
    match logits.iter().position(|x| !x.is_finite()) {
        Some(expert) => Err(RouteError::NonFiniteLogit { token, expert, value: logits[expert] }),
        None => Ok(()),
    }
}

/// Gate weights for already-selected experts.
///
/// * `SoftmaxThenTopK`: each selected expert's probability under a softmax
///   over all experts.
/// * `TopKThenSoftmax`: a softmax over only the selected logits.
fn gates_for(logits: &[f32], selected: &[usize], mode: GateMode) -> Vec<f32> {
    match mode {
        GateMode::SoftmaxThenTopK => {
            let probs = softmax(logits);
            selected.iter().map(|&e| probs[e]).collect()
        }
        GateMode::TopKThenSoftmax => {
            let picked: Vec<f32> = selected.iter().map(|&e| logits[e]).collect();
            softmax(&picked)
        }
    }
}

/// Full routing decision for one token: the top-`k` experts and their gates,
/// best expert first.
///
/// `token` is only used to make error messages point at the bad row.
fn route_one(
    logits: &[f32],
    k: usize,
    mode: GateMode,
    token: usize,
) -> Result<Vec<Route>, RouteError> {
    check_finite(logits, token)?;
    let selected = top_k_route(logits, k);
    let gates = gates_for(logits, &selected, mode);
    Ok(selected
        .into_iter()
        .zip(gates)
        .map(|(e, gate)| Route { expert_id: e as u32, gate })
        .collect())
}

/// Validates a flat `[num_tokens, num_experts]` logits batch for top-`k`
/// routing (shape, `k`, and that every logit is finite) and returns
/// `num_tokens`. Shared by the CPU and GPU entry points so both reject the
/// same inputs.
pub fn validate_batch(logits: &[f32], num_experts: usize, k: usize) -> Result<usize, RouteError> {
    check_shape(num_experts, k)?;
    if logits.len() % num_experts != 0 {
        return Err(RouteError::RaggedLogits { len: logits.len(), num_experts });
    }
    for (token, row) in logits.chunks_exact(num_experts).enumerate() {
        check_finite(row, token)?;
    }
    Ok(logits.len() / num_experts)
}

/// Routes a single token given its logits over all experts.
pub fn route_token(logits: &[f32], k: usize, mode: GateMode) -> Result<Vec<Route>, RouteError> {
    check_shape(logits.len(), k)?;
    route_one(logits, k, mode, 0)
}

/// Routes a batch of tokens.
///
/// `logits` is row-major `[num_tokens, num_experts]` (exactly what an MoE
/// gate `Linear` produces after flattening batch and sequence dims). The
/// result is flat with layout `[token * k + j]`, where `j = 0` is the
/// token's highest-scoring expert. That layout is shared with the GPU kernel.
pub fn route_batch(
    logits: &[f32],
    num_experts: usize,
    k: usize,
    mode: GateMode,
) -> Result<Vec<Route>, RouteError> {
    let num_tokens = validate_batch(logits, num_experts, k)?;
    let mut routes = Vec::with_capacity(num_tokens * k);
    for (token, row) in logits.chunks_exact(num_experts).enumerate() {
        routes.extend(route_one(row, k, mode, token)?);
    }
    Ok(routes)
}
