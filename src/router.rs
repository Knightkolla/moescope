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

/// How router logits become an expert choice and gate weights.
///
/// The two softmax modes select the same experts (softmax is monotonic, and
/// they select on raw logits); they differ only in the weights. `Sigmoid`
/// scores each expert independently and is the only mode that supports a
/// selection bias and group-limited routing (see [`Gating`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateMode {
    /// Softmax over *all* experts, then keep the top-k probabilities as-is.
    /// Gates sum to <= 1. Used by Switch Transformer / OLMoE
    /// (`norm_topk_prob = False`).
    #[default]
    SoftmaxThenTopK,
    /// Pick the top-k logits, then softmax over just those k, so gates are
    /// renormalized and sum to 1. Used by Mixtral, Granite, and Qwen-MoE
    /// with `norm_topk_prob = True`.
    TopKThenSoftmax,
    /// Score every expert with `sigmoid(logit)`, select on score (+ optional
    /// per-expert bias), gate with the unbiased score. Used by DeepSeek-V3,
    /// Kimi K2, GLM-4.5 (with bias and groups) and Llama 4 (plain).
    Sigmoid,
}

/// Most expert groups the group-limited router supports (one bit each in a
/// `u32` mask on the GPU). DeepSeek-V3 uses 8.
pub const MAX_GROUPS: usize = 32;

/// DeepSeek-V3-style group-limited routing: experts are split into
/// `num_groups` equal contiguous groups, each group is scored by the sum of
/// its two best (biased) expert scores, and experts are chosen only from the
/// `top_groups` best groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupLimit {
    pub num_groups: usize,
    pub top_groups: usize,
}

/// Full gating configuration: the mode plus the post-processing some model
/// families apply to the selected gates.
///
/// Gate pipeline: mode-specific weights, then (if `normalize`) divide by
/// their sum + 1e-20, then multiply by `scale`. That matches
/// `norm_topk_prob` and `routed_scaling_factor` in Hugging Face configs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gating {
    pub mode: GateMode,
    pub normalize: bool,
    pub scale: f32,
    /// Sigmoid mode only.
    pub groups: Option<GroupLimit>,
}

impl Default for Gating {
    fn default() -> Self {
        GateMode::default().into()
    }
}

impl From<GateMode> for Gating {
    fn from(mode: GateMode) -> Self {
        Gating {
            mode,
            normalize: false,
            scale: 1.0,
            groups: None,
        }
    }
}

impl Gating {
    /// Checks this configuration (and an optional per-expert selection
    /// bias) against a layer with `num_experts` experts routing to `k`.
    pub fn validate(
        &self,
        num_experts: usize,
        k: usize,
        bias: Option<&[f32]>,
    ) -> Result<(), RouteError> {
        let bad = |msg: String| Err(RouteError::InvalidGating(msg));
        if !self.scale.is_finite() {
            return bad(format!("scale must be finite, got {}", self.scale));
        }
        if let Some(b) = bias {
            if self.mode != GateMode::Sigmoid {
                return bad("a selection bias needs mode 'sigmoid'".into());
            }
            if b.len() != num_experts {
                return bad(format!(
                    "bias has {} entries, layer has {num_experts} experts",
                    b.len()
                ));
            }
            if let Some(i) = b.iter().position(|x| !x.is_finite()) {
                return bad(format!("bias[{i}] = {} is not finite", b[i]));
            }
        }
        if let Some(GroupLimit {
            num_groups,
            top_groups,
        }) = self.groups
        {
            if self.mode != GateMode::Sigmoid {
                return bad("group-limited routing needs mode 'sigmoid'".into());
            }
            if num_groups == 0 || num_groups > MAX_GROUPS {
                return bad(format!(
                    "num_groups must be 1..={MAX_GROUPS}, got {num_groups}"
                ));
            }
            if !num_experts.is_multiple_of(num_groups) {
                return bad(format!(
                    "{num_experts} experts do not split into {num_groups} equal groups"
                ));
            }
            let group_size = num_experts / num_groups;
            if num_groups > 1 && group_size < 2 {
                return bad("groups need at least 2 experts each".into());
            }
            if top_groups == 0 || top_groups > num_groups {
                return bad(format!(
                    "top_groups must be 1..={num_groups}, got {top_groups}"
                ));
            }
            if top_groups * group_size < k {
                return bad(format!(
                    "k = {k} exceeds the {} experts in the top {top_groups} groups",
                    top_groups * group_size
                ));
            }
        }
        Ok(())
    }
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
    NonFiniteLogit {
        token: usize,
        expert: usize,
        value: f32,
    },
    /// The [`Gating`] configuration (or its bias) does not fit the layer.
    InvalidGating(String),
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
            RouteError::NonFiniteLogit {
                token,
                expert,
                value,
            } => write!(
                f,
                "non-finite logit {value} at token {token}, expert {expert}"
            ),
            RouteError::InvalidGating(msg) => write!(f, "invalid gating: {msg}"),
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
        Some(expert) => Err(RouteError::NonFiniteLogit {
            token,
            expert,
            value: logits[expert],
        }),
        None => Ok(()),
    }
}

/// Logistic sigmoid, the per-expert score of [`GateMode::Sigmoid`].
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid-mode selection scores: `sigmoid(logit) + bias`, with experts
/// outside the best `top_groups` groups set to -inf when groups are on.
/// Groups are ranked by the sum of their two best scores; ties go to the
/// lower group index.
fn sigmoid_choice_scores(logits: &[f32], gating: &Gating, bias: Option<&[f32]>) -> Vec<f32> {
    let mut scores: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(e, &x)| sigmoid(x) + bias.map_or(0.0, |b| b[e]))
        .collect();
    if let Some(GroupLimit {
        num_groups,
        top_groups,
    }) = gating.groups
        && num_groups > 1
    {
        let size = scores.len() / num_groups;
        let group_scores: Vec<f32> = scores
            .chunks_exact(size)
            .map(|g| top_k_route(g, 2).iter().map(|&i| g[i]).sum())
            .collect();
        let kept = top_k_route(&group_scores, top_groups);
        for (g, chunk) in scores.chunks_exact_mut(size).enumerate() {
            if !kept.contains(&g) {
                chunk.fill(f32::NEG_INFINITY);
            }
        }
    }
    scores
}

/// Selected experts and their raw (pre-normalization) gate weights.
///
/// * `SoftmaxThenTopK`: each selected expert's probability under a softmax
///   over all experts.
/// * `TopKThenSoftmax`: a softmax over only the selected logits.
/// * `Sigmoid`: select on the (biased, group-masked) sigmoid scores, gate
///   with the plain sigmoid of the logit.
fn select_and_gate(
    logits: &[f32],
    k: usize,
    gating: &Gating,
    bias: Option<&[f32]>,
) -> (Vec<usize>, Vec<f32>) {
    match gating.mode {
        GateMode::SoftmaxThenTopK => {
            let selected = top_k_route(logits, k);
            let probs = softmax(logits);
            let gates = selected.iter().map(|&e| probs[e]).collect();
            (selected, gates)
        }
        GateMode::TopKThenSoftmax => {
            let selected = top_k_route(logits, k);
            let picked: Vec<f32> = selected.iter().map(|&e| logits[e]).collect();
            (selected, softmax(&picked))
        }
        GateMode::Sigmoid => {
            let selected = top_k_route(&sigmoid_choice_scores(logits, gating, bias), k);
            let gates = selected.iter().map(|&e| sigmoid(logits[e])).collect();
            (selected, gates)
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
    gating: &Gating,
    bias: Option<&[f32]>,
    token: usize,
) -> Result<Vec<Route>, RouteError> {
    check_finite(logits, token)?;
    let (selected, mut gates) = select_and_gate(logits, k, gating, bias);
    if gating.normalize {
        let sum: f32 = gates.iter().sum::<f32>() + 1e-20;
        gates.iter_mut().for_each(|g| *g /= sum);
    }
    Ok(selected
        .into_iter()
        .zip(gates)
        .map(|(e, gate)| Route {
            expert_id: e as u32,
            gate: gate * gating.scale,
        })
        .collect())
}

/// Validates a flat `[num_tokens, num_experts]` logits batch for top-`k`
/// routing (shape, `k`, and that every logit is finite) and returns
/// `num_tokens`. Shared by the CPU and GPU entry points so both reject the
/// same inputs.
pub fn validate_batch(logits: &[f32], num_experts: usize, k: usize) -> Result<usize, RouteError> {
    check_shape(num_experts, k)?;
    if !logits.len().is_multiple_of(num_experts) {
        return Err(RouteError::RaggedLogits {
            len: logits.len(),
            num_experts,
        });
    }
    for (token, row) in logits.chunks_exact(num_experts).enumerate() {
        check_finite(row, token)?;
    }
    Ok(logits.len() / num_experts)
}

/// Routes a single token given its logits over all experts.
pub fn route_token(logits: &[f32], k: usize, mode: GateMode) -> Result<Vec<Route>, RouteError> {
    check_shape(logits.len(), k)?;
    route_one(logits, k, &mode.into(), None, 0)
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
    route_batch_with(logits, num_experts, k, &mode.into(), None)
}

/// [`route_batch`] with a full [`Gating`] configuration and an optional
/// per-expert selection `bias` (sigmoid mode; DeepSeek-V3's
/// `e_score_correction_bias`). Same flat `[token * k + j]` layout.
pub fn route_batch_with(
    logits: &[f32],
    num_experts: usize,
    k: usize,
    gating: &Gating,
    bias: Option<&[f32]>,
) -> Result<Vec<Route>, RouteError> {
    let num_tokens = validate_batch(logits, num_experts, k)?;
    gating.validate(num_experts, k, bias)?;
    let mut routes = Vec::with_capacity(num_tokens * k);
    for (token, row) in logits.chunks_exact(num_experts).enumerate() {
        routes.extend(route_one(row, k, gating, bias, token)?);
    }
    Ok(routes)
}

#[cfg(test)]
mod gating_tests {
    use super::*;

    fn sig(groups: Option<(usize, usize)>) -> Gating {
        Gating {
            mode: GateMode::Sigmoid,
            normalize: false,
            scale: 1.0,
            groups: groups.map(|(num_groups, top_groups)| GroupLimit {
                num_groups,
                top_groups,
            }),
        }
    }

    #[test]
    fn plain_sigmoid_selects_like_logits_and_gates_with_sigmoid() {
        let x = [0.5, -1.0, 3.0, 2.0];
        let r = route_batch_with(&x, 4, 2, &sig(None), None).unwrap();
        assert_eq!((r[0].expert_id, r[1].expert_id), (2, 3));
        assert!((r[0].gate - sigmoid(3.0)).abs() < 1e-7);
    }

    #[test]
    fn bias_moves_selection_but_not_gates() {
        let x = [0.5, -1.0, 3.0, 2.0];
        let bias = [0.0, 5.0, 0.0, 0.0];
        let r = route_batch_with(&x, 4, 1, &sig(None), Some(&bias)).unwrap();
        assert_eq!(r[0].expert_id, 1);
        assert!((r[0].gate - sigmoid(-1.0)).abs() < 1e-7);
    }

    #[test]
    fn groups_limit_the_choice() {
        // Group 0 = experts 0..4, group 1 = 4..8. Group 1 has the best single
        // expert (7) but group 0 has the better top-2 sum, so with one group
        // kept, expert 7 is excluded.
        let x = [2.0, 2.0, -5.0, -5.0, -5.0, -5.0, -5.0, 3.0];
        let r = route_batch_with(&x, 8, 2, &sig(Some((2, 1))), None).unwrap();
        assert_eq!((r[0].expert_id, r[1].expert_id), (0, 1));
        let all = route_batch_with(&x, 8, 2, &sig(Some((2, 2))), None).unwrap();
        assert_eq!(all[0].expert_id, 7);
    }

    #[test]
    fn normalize_and_scale() {
        let g = Gating {
            normalize: true,
            scale: 2.5,
            ..sig(None)
        };
        let r = route_batch_with(&[1.0, 0.0, -1.0], 3, 2, &g, None).unwrap();
        assert!(((r[0].gate + r[1].gate) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_configs() {
        let x = [0.0; 16];
        let err = |g: Gating, b: Option<&[f32]>| route_batch_with(&x, 16, 4, &g, b).is_err();
        assert!(err(sig(Some((3, 1))), None));
        assert!(err(sig(Some((16, 1))), None)); // groups of 1 expert
        assert!(err(sig(Some((4, 5))), None));
        assert!(err(sig(Some((8, 1))), None)); // 2 experts kept < k = 4
        assert!(err(sig(None), Some(&[0.0; 3])));
        assert!(err(GateMode::TopKThenSoftmax.into(), Some(&[0.0; 16])));
        assert!(!err(sig(Some((4, 2))), Some(&[0.1; 16])));
    }
}
