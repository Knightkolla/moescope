//! Phase 3: the GPU router must agree with the CPU router exactly on expert
//! ids and within 1e-5 on gates.
#![cfg(feature = "cuda")]

mod common;

use common::{gpu, logits};
use moescope::gpu::{GpuError, kernels::MAX_K};
use moescope::router::{GateMode, RouteError, route_batch};

const MODES: [GateMode; 2] = [GateMode::SoftmaxThenTopK, GateMode::TopKThenSoftmax];

fn assert_matches_cpu(logits: &[f32], num_experts: usize, k: usize, mode: GateMode) {
    let Some(router) = gpu() else { return };
    let cpu = route_batch(logits, num_experts, k, mode).unwrap();
    let gpu = router.route(logits, num_experts, k, mode).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        let (t, j) = (i / k, i % k);
        assert_eq!(c.expert_id, g.expert_id, "expert mismatch at token {t}, slot {j} ({mode:?}, E={num_experts}, k={k})");
        assert!((c.gate - g.gate).abs() <= 1e-5, "gate mismatch at token {t}, slot {j}: cpu {} gpu {}", c.gate, g.gate);
    }
}

#[test]
fn matches_cpu_across_shapes_and_modes() {
    let shapes = [(1, 1, 1), (7, 8, 2), (300, 8, 2), (513, 64, 8), (5000, 128, 4), (64, 16, 16), (40, 64, MAX_K)];
    for (seed, &(tokens, experts, k)) in shapes.iter().enumerate() {
        let x = logits(tokens, experts, seed as u64);
        for mode in MODES {
            assert_matches_cpu(&x, experts, k, mode);
        }
    }
}

#[test]
fn ties_match_cpu() {
    // Heavy ties: logits quantized to a few values.
    let x: Vec<f32> = logits(257, 16, 99).iter().map(|v| v.round()).collect();
    for mode in MODES {
        assert_matches_cpu(&x, 16, 4, mode);
    }
    assert_matches_cpu(&vec![0.25; 8 * 32], 32, 5, GateMode::TopKThenSoftmax);
}

#[test]
fn rejects_what_cpu_rejects() {
    let Some(router) = gpu() else { return };
    let mode = GateMode::SoftmaxThenTopK;
    assert!(matches!(
        router.route(&[1.0, 2.0], 2, 3, mode),
        Err(GpuError::Route(RouteError::KExceedsExperts { k: 3, num_experts: 2 }))
    ));
    assert!(matches!(router.route(&[1.0, f32::NAN], 2, 1, mode), Err(GpuError::Route(RouteError::NonFiniteLogit { .. }))));
    let wide = vec![0.0; 64];
    assert!(matches!(router.route(&wide, 64, MAX_K + 1, mode), Err(GpuError::Limit(_))));
    assert!(router.route(&[], 4, 2, mode).unwrap().is_empty());
}
