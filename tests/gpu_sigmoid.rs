//! Sigmoid gating (DeepSeek-V3 / GLM-4.5 / Llama 4) on the GPU must agree
//! with the CPU reference router: same experts, gates within 1e-5, including
//! the selection bias and group-limited routing.
#![cfg(feature = "cuda")]

mod common;

use common::{gpu, logits};
use moescope::gpu::{GpuError, GpuGate};
use moescope::router::{GateMode, Gating, GroupLimit, RouteError, route_batch_with};

fn sigmoid_gating(normalize: bool, scale: f32, groups: Option<(usize, usize)>) -> Gating {
    Gating {
        mode: GateMode::Sigmoid,
        normalize,
        scale,
        groups: groups.map(|(num_groups, top_groups)| GroupLimit {
            num_groups,
            top_groups,
        }),
    }
}

fn assert_matches_cpu(x: &[f32], e: usize, k: usize, gating: Gating, bias: Option<&[f32]>) {
    let Some(router) = gpu() else { return };
    let cpu = route_batch_with(x, e, k, &gating, bias).unwrap();
    let dev_bias = bias.map(|b| router.upload_bias(b).unwrap());
    let gate = GpuGate {
        gating,
        bias: dev_bias.as_ref(),
    };
    let gpu = router.route(x, e, k, gate).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert_eq!(
            c.expert_id,
            g.expert_id,
            "expert mismatch at token {}, slot {} ({gating:?}, E={e}, k={k})",
            i / k,
            i % k
        );
        assert!(
            (c.gate - g.gate).abs() <= 1e-5 * gating.scale.abs().max(1.0),
            "gate mismatch at {i}: cpu {} gpu {}",
            c.gate,
            g.gate
        );
    }
}

#[test]
fn plain_sigmoid_matches_cpu() {
    for (seed, &(n, e, k)) in [(1, 1, 1), (300, 8, 2), (513, 64, 8), (2000, 256, 8)]
        .iter()
        .enumerate()
    {
        let x = logits(n, e, seed as u64 + 10);
        assert_matches_cpu(&x, e, k, sigmoid_gating(false, 1.0, None), None);
        assert_matches_cpu(&x, e, k, sigmoid_gating(true, 2.5, None), None);
    }
}

#[test]
fn bias_and_groups_match_cpu() {
    // DeepSeek-V3 shape (256 experts, 8 groups, top 4 groups, k = 8) and a
    // few smaller ones, each with a non-trivial bias.
    for (seed, &(n, e, k, groups, top)) in [
        (1000, 256, 8, 8, 4),
        (777, 64, 6, 8, 3),
        (500, 32, 4, 4, 2),
        (300, 16, 4, 2, 1),
        (128, 64, 8, 32, 32),
    ]
    .iter()
    .enumerate()
    {
        let x = logits(n, e, seed as u64 + 50);
        let bias: Vec<f32> = logits(1, e, seed as u64 + 90)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        for normalize in [false, true] {
            let g = sigmoid_gating(normalize, 2.5, Some((groups, top)));
            assert_matches_cpu(&x, e, k, g, Some(&bias));
            assert_matches_cpu(&x, e, k, g, None);
        }
    }
}

#[test]
fn bias_changes_the_choice() {
    // A big bias on expert 3 must pull it into every token's top-k.
    let Some(router) = gpu() else { return };
    let (n, e, k) = (64, 16, 2);
    let x = logits(n, e, 7);
    let mut bias = vec![0.0; e];
    bias[3] = 10.0;
    let dev = router.upload_bias(&bias).unwrap();
    let gate = GpuGate {
        gating: sigmoid_gating(false, 1.0, None),
        bias: Some(&dev),
    };
    let routes = router.route(&x, e, k, gate).unwrap();
    for t in 0..n {
        assert_eq!(routes[t * k].expert_id, 3);
        // ...while its gate is still the *unbiased* sigmoid score.
        let s = 1.0 / (1.0 + (-x[t * e + 3]).exp());
        assert!((routes[t * k].gate - s).abs() < 1e-6);
    }
}

#[test]
fn invalid_gating_is_rejected() {
    let Some(router) = gpu() else { return };
    let x = logits(4, 16, 1);
    let bad = |g: Gating| {
        matches!(
            router.route(&x, 16, 4, g),
            Err(GpuError::Route(RouteError::InvalidGating(_)))
        )
    };
    assert!(bad(sigmoid_gating(false, 1.0, Some((3, 1))))); // 16 % 3 != 0
    assert!(bad(sigmoid_gating(false, 1.0, Some((4, 0)))));
    assert!(bad(sigmoid_gating(false, 1.0, Some((8, 1))))); // only 2 experts kept < k
    assert!(bad(Gating {
        groups: Some(GroupLimit {
            num_groups: 2,
            top_groups: 1
        }),
        ..GateMode::SoftmaxThenTopK.into()
    }));
    let dev = router.upload_bias(&[0.0; 8]).unwrap();
    let wrong_len = GpuGate {
        gating: sigmoid_gating(false, 1.0, None),
        bias: Some(&dev),
    };
    assert!(matches!(
        router.route(&x, 16, 4, wrong_len),
        Err(GpuError::Route(RouteError::InvalidGating(_)))
    ));
}

#[test]
fn log_only_device_path_uses_bias() {
    // The zero-copy logging path (what Python uses) with groups + bias must
    // log exactly what the CPU router decides.
    use moescope::gpu::TelemetryBuffer;
    let Some(router) = gpu() else { return };
    let (n, e, k) = (333, 64, 8);
    let x = logits(n, e, 21);
    let bias: Vec<f32> = logits(1, e, 22).iter().map(|v| v * 0.05).collect();
    let g = sigmoid_gating(true, 2.5, Some((8, 4)));
    let cpu = route_batch_with(&x, e, k, &g, Some(&bias)).unwrap();
    let dev = router.upload_bias(&bias).unwrap();
    let mut log = TelemetryBuffer::new(router, n * k).unwrap();
    router
        .log_only(
            &x,
            e,
            k,
            GpuGate {
                gating: g,
                bias: Some(&dev),
            },
            9,
            0,
            &mut log,
        )
        .unwrap();
    let mut recs = log.drain().unwrap().records;
    recs.sort_by_key(|r| r.token_id); // stable: keeps best-first order within a token
    for (i, (c, r)) in cpu.iter().zip(&recs).enumerate() {
        assert_eq!(
            (c.expert_id as u16, i as u32 / k as u32),
            (r.expert_id, r.token_id)
        );
        assert!((c.gate - r.gate).abs() < 3e-5);
        assert_eq!(r.layer_id, 9);
    }
}
