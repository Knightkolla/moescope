//! Phase 4 (CPU): dispatch plan, load histogram, and bounded telemetry log.

use moescope::dispatcher::dispatch;
use moescope::router::{GateMode, Route, route_batch};
use moescope::telemetry::{HostTelemetry, TelemetryError, TelemetryRecord, records_for};
use proptest::prelude::*;

fn r(expert_id: u32, gate: f32) -> Route {
    Route { expert_id, gate }
}

#[test]
fn dispatch_groups_assignments_by_expert_in_token_order() {
    // 3 tokens, k = 2, 4 experts.
    let routes = [
        r(1, 0.6),
        r(3, 0.4),
        r(1, 0.7),
        r(0, 0.3),
        r(2, 0.5),
        r(1, 0.5),
    ];
    let plan = dispatch(&routes, 2, 4);
    assert_eq!(plan.expert_load(), &[1, 3, 1, 1]);
    assert_eq!(plan.assignments(0), &[3]);
    assert_eq!(plan.assignments(1), &[0, 2, 5]); // tokens 0, 1, 2
    assert_eq!(plan.assignments(2), &[4]);
    assert_eq!(plan.assignments(3), &[1]);
}

#[test]
fn records_carry_token_layer_expert_gate() {
    let routes = [r(5, 0.9), r(2, 0.1), r(7, 1.0), r(0, 0.0)];
    let recs = records_for(&routes, 2, 3, 100).unwrap();
    assert_eq!(
        recs[0],
        TelemetryRecord {
            token_id: 100,
            layer_id: 3,
            flags: 0,
            expert_id: 5,
            gate: 0.9
        }
    );
    assert_eq!(
        recs[3],
        TelemetryRecord {
            token_id: 101,
            layer_id: 3,
            flags: 0,
            expert_id: 0,
            gate: 0.0
        }
    );
}

#[test]
fn id_overflow_is_reported() {
    assert_eq!(
        records_for(&[r(70_000, 1.0)], 1, 0, 0),
        Err(TelemetryError::ExpertIdOverflow { expert_id: 70_000 })
    );
    assert!(matches!(
        records_for(&[r(0, 1.0), r(0, 1.0)], 1, 0, u32::MAX),
        Err(TelemetryError::TokenIdOverflow { .. })
    ));
    // The last representable token id is fine.
    assert!(records_for(&[r(0, 1.0)], 1, 0, u32::MAX).is_ok());
}

#[test]
fn overflow_is_counted_not_lost_silently() {
    let mut log = HostTelemetry::new(5);
    let routes = [r(0, 0.5); 4]; // 2 tokens x k=2
    log.log_routes(&routes, 2, 0, 0).unwrap();
    log.log_routes(&routes, 2, 1, 2).unwrap(); // 8 records into capacity 5
    assert_eq!(log.len(), 5);
    let d = log.drain();
    assert_eq!((d.records.len(), d.dropped), (5, 3));
    assert!(log.is_empty(), "drain resets the log");
    assert_eq!(log.drain().dropped, 0);
}

proptest! {
    #[test]
    fn dispatch_is_a_partition_of_all_assignments(
        (e, k, logits) in (1usize..=16, 1usize..=6)
            .prop_flat_map(|(e, t)| (Just(e), 1..=e, prop::collection::vec(-5.0f32..5.0, e * t)))
    ) {
        let routes = route_batch(&logits, e, k, GateMode::TopKThenSoftmax).unwrap();
        let plan = dispatch(&routes, k, e);
        prop_assert_eq!(plan.expert_load().iter().sum::<u32>() as usize, routes.len());
        let mut seen = vec![false; routes.len()];
        for x in 0..e {
            for &a in plan.assignments(x) {
                prop_assert_eq!(routes[a as usize].expert_id as usize, x);
                prop_assert!(!seen[a as usize]);
                seen[a as usize] = true;
            }
        }
        prop_assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn kept_plus_dropped_equals_logged(cap in 1usize..50, batches in 1usize..6, tokens in 1usize..8) {
        let mut log = HostTelemetry::new(cap);
        let routes = vec![r(1, 0.5); tokens * 2];
        for b in 0..batches {
            log.log_routes(&routes, 2, b as u8, 0).unwrap();
        }
        let d = log.drain();
        let total = batches * tokens * 2;
        prop_assert_eq!(d.records.len(), total.min(cap));
        prop_assert_eq!(d.records.len() as u64 + d.dropped, total as u64);
    }
}

#[test]
fn kept_mask_flags_decisions_the_experts_never_computed() {
    use moescope::telemetry::{CAPACITY_DROP, KeptMask, records_for_masked};
    // 2 tokens, k = 2, 3 experts. Token 0 -> {2, 0}, token 1 -> {1, 2}.
    let routes = [r(2, 0.6), r(0, 0.4), r(1, 0.7), r(2, 0.3)];
    // Expert 2 was full when token 1 arrived; everything else computed.
    let mask = [1, 0, 1, /* token 1 */ 0, 1, 0];
    let kept = KeptMask {
        mask: &mask,
        num_experts: 3,
    };
    let recs = records_for_masked(&routes, 2, 0, 0, Some(kept)).unwrap();
    let flags: Vec<u8> = recs.iter().map(|x| x.flags).collect();
    assert_eq!(flags, [0, 0, 0, CAPACITY_DROP]);
    assert_eq!(recs[3].gate, 0.3, "gate stays the router's");

    let short = KeptMask {
        mask: &mask[..5],
        num_experts: 3,
    };
    assert!(matches!(
        records_for_masked(&routes, 2, 0, 0, Some(short)),
        Err(TelemetryError::KeptMaskShape {
            expected: 6,
            found: 5
        })
    ));
}
