//! Phase 4 (GPU): fused route + dispatch + telemetry kernel vs the CPU path.
#![cfg(feature = "cuda")]

mod common;

use common::{gpu, logits};
use moescope::dispatcher::dispatch;
use moescope::gpu::TelemetryBuffer;
use moescope::router::{GateMode, route_batch};
use moescope::telemetry::{HostTelemetry, TelemetryRecord};

/// GPU append order is nondeterministic; compare logs as sorted multisets.
fn sorted(mut recs: Vec<TelemetryRecord>) -> Vec<TelemetryRecord> {
    recs.sort_by_key(|r| (r.layer_id, r.token_id, r.expert_id));
    recs
}

#[test]
fn routes_dispatch_and_histogram_match_cpu() {
    let Some(router) = gpu() else { return };
    for (seed, &(n, e, k)) in [(1usize, 4usize, 1usize), (300, 8, 2), (2049, 64, 8), (100, 16, 16)].iter().enumerate() {
        let x = logits(n, e, 100 + seed as u64);
        for mode in [GateMode::SoftmaxThenTopK, GateMode::TopKThenSoftmax] {
            let mut log = TelemetryBuffer::new(router, n * k).unwrap();
            let dev = router.route_dispatch_log(&x, e, k, mode, 0, 0, &mut log).unwrap();
            let (routes, plan) = dev.to_host().unwrap();
            let cpu_routes = route_batch(&x, e, k, mode).unwrap();
            let cpu_plan = dispatch(&cpu_routes, k, e);
            for (c, g) in cpu_routes.iter().zip(&routes) {
                assert_eq!(c.expert_id, g.expert_id);
                assert!((c.gate - g.gate).abs() <= 1e-5);
            }
            assert_eq!(plan.expert_load(), cpu_plan.expert_load(), "histogram n={n} e={e} k={k}");
            assert_eq!(dev.expert_load().unwrap(), cpu_plan.expert_load());
            for ex in 0..e {
                let mut got = plan.assignments(ex).to_vec();
                got.sort();
                assert_eq!(got, cpu_plan.assignments(ex), "expert {ex} token group");
            }
        }
    }
}

#[test]
fn telemetry_accumulates_across_layers_and_drains_once() {
    let Some(router) = gpu() else { return };
    let (n, e, k, layers) = (777, 32, 4, 3u16);
    let mut gpu_log = TelemetryBuffer::new(router, n * k * layers as usize).unwrap();
    let mut cpu_log = HostTelemetry::new(n * k * layers as usize);
    for layer in 0..layers {
        let x = logits(n, e, layer as u64);
        router.route_dispatch_log(&x, e, k, GateMode::SoftmaxThenTopK, layer, 1000, &mut gpu_log).unwrap();
        let routes = route_batch(&x, e, k, GateMode::SoftmaxThenTopK).unwrap();
        cpu_log.log_routes(&routes, k, layer, 1000).unwrap();
    }
    let got = gpu_log.drain().unwrap();
    let want = cpu_log.drain();
    assert_eq!(got.dropped, 0);
    let (got, want) = (sorted(got.records), sorted(want.records));
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!((g.token_id, g.layer_id, g.expert_id), (w.token_id, w.layer_id, w.expert_id));
        assert!((g.gate - w.gate).abs() <= 1e-5);
    }
    let again = gpu_log.drain().unwrap();
    assert!(again.records.is_empty() && again.dropped == 0, "drain must reset the buffer");
}

#[test]
fn overflow_counts_dropped_records() {
    let Some(router) = gpu() else { return };
    let (n, e, k, cap) = (1000, 8, 2, 1500);
    let mut log = TelemetryBuffer::new(router, cap).unwrap();
    let x = logits(n, e, 5);
    router.route_dispatch_log(&x, e, k, GateMode::TopKThenSoftmax, 0, 0, &mut log).unwrap();
    router.route_dispatch_log(&x, e, k, GateMode::TopKThenSoftmax, 1, 0, &mut log).unwrap();
    let d = log.drain().unwrap();
    assert_eq!(d.records.len(), cap);
    assert_eq!(d.dropped, (2 * n * k - cap) as u64);
    // Every kept record is a real decision (no torn/zeroed slots).
    assert!(d.records.iter().all(|r| r.gate > 0.0 && (r.expert_id as usize) < e && (r.token_id as usize) < n));
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(router) = gpu() else { return };
    let mut log = TelemetryBuffer::new(router, 4).unwrap();
    let dev = router.route_dispatch_log(&[], 8, 2, GateMode::SoftmaxThenTopK, 0, 0, &mut log).unwrap();
    let (routes, plan) = dev.to_host().unwrap();
    assert!(routes.is_empty() && plan.expert_load().iter().all(|&c| c == 0));
    assert!(log.drain().unwrap().records.is_empty());
}
