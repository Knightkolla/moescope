//! Phase 1: softmax + gate weights for the CPU router.

use moescope::router::{GateMode, Route, RouteError, route_batch, route_token, softmax, top_k_route};
use proptest::prelude::*;

const MODES: [GateMode; 2] = [GateMode::SoftmaxThenTopK, GateMode::TopKThenSoftmax];

#[test]
fn softmax_matches_hand_computation() {
    let p = softmax(&[0.0, (2.0f32).ln()]);
    assert!((p[0] - 1.0 / 3.0).abs() < 1e-6);
    assert!((p[1] - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn softmax_is_stable_for_huge_logits() {
    let p = softmax(&[1000.0, 1000.0]);
    assert_eq!(p, vec![0.5, 0.5]);
}

#[test]
fn softmax_then_topk_keeps_full_softmax_probs() {
    let logits = [1.0, 3.0, 2.0, 0.0];
    let probs = softmax(&logits);
    let routes = route_token(&logits, 2, GateMode::SoftmaxThenTopK).unwrap();
    assert_eq!(routes, vec![
        Route { expert_id: 1, gate: probs[1] },
        Route { expert_id: 2, gate: probs[2] },
    ]);
}

#[test]
fn topk_then_softmax_renormalizes_over_selected() {
    // Mixtral-style: softmax over [3, 2] only.
    let routes = route_token(&[1.0, 3.0, 2.0, 0.0], 2, GateMode::TopKThenSoftmax).unwrap();
    let e = (1.0f32).exp();
    assert_eq!(routes[0].expert_id, 1);
    assert_eq!(routes[1].expert_id, 2);
    assert!((routes[0].gate - e / (e + 1.0)).abs() < 1e-6);
    assert!((routes[1].gate - 1.0 / (e + 1.0)).abs() < 1e-6);
}

#[test]
fn batch_layout_is_token_major() {
    let logits = [0.1, 0.9, 0.3, 0.7, /* token 1 */ 0.5, 0.2, 0.8, 0.4];
    let routes = route_batch(&logits, 4, 2, GateMode::TopKThenSoftmax).unwrap();
    let ids: Vec<u32> = routes.iter().map(|r| r.expert_id).collect();
    assert_eq!(ids, vec![1, 3, 2, 0]); // [token*k + j]
}

#[test]
fn ties_break_toward_lower_expert_index() {
    let routes = route_token(&[0.5, 1.0, 1.0, 1.0], 2, GateMode::SoftmaxThenTopK).unwrap();
    assert_eq!(routes[0].expert_id, 1);
    assert_eq!(routes[1].expert_id, 2);
}

#[test]
fn k_greater_than_num_experts_is_an_error() {
    assert_eq!(
        route_token(&[1.0, 2.0], 3, GateMode::SoftmaxThenTopK),
        Err(RouteError::KExceedsExperts { k: 3, num_experts: 2 })
    );
    assert_eq!(
        route_batch(&[1.0, 2.0], 2, 3, GateMode::TopKThenSoftmax),
        Err(RouteError::KExceedsExperts { k: 3, num_experts: 2 })
    );
}

#[test]
fn shape_errors() {
    assert_eq!(route_token(&[1.0], 0, GateMode::SoftmaxThenTopK), Err(RouteError::ZeroK));
    assert_eq!(route_batch(&[], 0, 1, GateMode::SoftmaxThenTopK), Err(RouteError::NoExperts));
    assert_eq!(
        route_batch(&[1.0, 2.0, 3.0], 2, 1, GateMode::SoftmaxThenTopK),
        Err(RouteError::RaggedLogits { len: 3, num_experts: 2 })
    );
}

#[test]
fn nan_logit_is_an_error_not_a_panic() {
    let err = route_batch(&[1.0, 2.0, 0.0, f32::NAN], 2, 1, GateMode::SoftmaxThenTopK).unwrap_err();
    assert!(matches!(err, RouteError::NonFiniteLogit { token: 1, expert: 1, .. }));
}

#[test]
fn legacy_top_k_does_not_panic_on_nan() {
    assert_eq!(top_k_route(&[f32::NAN, 0.2, 0.9], 2), vec![2, 1]);
}

/// Strategy: (num_experts, k, flat logits for a few tokens).
fn batch_strategy() -> impl Strategy<Value = (usize, usize, Vec<f32>)> {
    (1usize..=16, 1usize..=4).prop_flat_map(|(e, tokens)| {
        (Just(e), 1..=e, prop::collection::vec(-20.0f32..20.0, e * tokens))
    })
}

proptest! {
    #[test]
    fn renormalized_gates_sum_to_one((e, k, logits) in batch_strategy()) {
        let routes = route_batch(&logits, e, k, GateMode::TopKThenSoftmax).unwrap();
        for token in routes.chunks_exact(k) {
            let sum: f32 = token.iter().map(|r| r.gate).sum();
            prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        }
    }

    #[test]
    fn softmax_then_topk_gates_sum_to_at_most_one((e, k, logits) in batch_strategy()) {
        let routes = route_batch(&logits, e, k, GateMode::SoftmaxThenTopK).unwrap();
        for token in routes.chunks_exact(k) {
            let sum: f32 = token.iter().map(|r| r.gate).sum();
            prop_assert!(sum <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn both_modes_select_the_same_distinct_experts_in_descending_order(
        (e, k, logits) in batch_strategy()
    ) {
        let a = route_batch(&logits, e, k, MODES[0]).unwrap();
        let b = route_batch(&logits, e, k, MODES[1]).unwrap();
        for (t, (ta, tb)) in a.chunks_exact(k).zip(b.chunks_exact(k)).enumerate() {
            let ids: Vec<u32> = ta.iter().map(|r| r.expert_id).collect();
            prop_assert_eq!(&ids, &tb.iter().map(|r| r.expert_id).collect::<Vec<_>>());
            let mut dedup = ids.clone();
            dedup.sort();
            dedup.dedup();
            prop_assert_eq!(dedup.len(), k);
            let row = &logits[t * e..(t + 1) * e];
            for w in ids.windows(2) {
                prop_assert!(row[w[0] as usize] >= row[w[1] as usize]);
            }
            for g in ta.iter().chain(tb) {
                prop_assert!(g.gate > 0.0 && g.gate <= 1.0);
            }
        }
    }

    #[test]
    fn ties_are_deterministic(e in 2usize..=16, k_frac in 0.0f64..1.0, v in -5.0f32..5.0) {
        // All logits equal: every mode must pick experts 0..k in order.
        let k = 1 + ((e - 1) as f64 * k_frac) as usize;
        for mode in MODES {
            let routes = route_token(&vec![v; e], k, mode).unwrap();
            let ids: Vec<u32> = routes.iter().map(|r| r.expert_id).collect();
            prop_assert_eq!(ids, (0..k as u32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn k_over_num_experts_always_errors(e in 1usize..=16, extra in 1usize..=4) {
        let logits = vec![0.0; e];
        for mode in MODES {
            prop_assert_eq!(
                route_token(&logits, e + extra, mode),
                Err(RouteError::KExceedsExperts { k: e + extra, num_experts: e })
            );
        }
    }

    #[test]
    fn any_nan_position_errors_without_panicking(e in 1usize..=16, pos_frac in 0.0f64..1.0) {
        let mut logits = vec![1.0; e];
        let pos = ((e - 1) as f64 * pos_frac) as usize;
        logits[pos] = f32::NAN;
        for mode in MODES {
            let is_nan_error = matches!(
                route_token(&logits, 1, mode),
                Err(RouteError::NonFiniteLogit { expert, .. }) if expert == pos
            );
            prop_assert!(is_nan_error);
        }
    }
}
