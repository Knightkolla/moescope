//! Shared helpers for the GPU integration tests.
#![allow(dead_code)]

use std::sync::OnceLock;

use moescope::gpu::{GpuRouter, cuda_available};

/// One router per test binary (NVRTC compile is the slow part).
///
/// Returns `None` (test should skip) when there is no CUDA driver, unless
/// `MOESCOPE_REQUIRE_GPU=1`, in which case a missing GPU is a hard failure:
/// use that on Colab so a misconfigured runtime can't pass silently.
pub fn gpu() -> Option<&'static GpuRouter> {
    static ROUTER: OnceLock<Option<GpuRouter>> = OnceLock::new();
    let required = std::env::var("MOESCOPE_REQUIRE_GPU").is_ok_and(|v| v == "1");
    let router = ROUTER.get_or_init(|| {
        if !cuda_available() {
            assert!(
                !required,
                "MOESCOPE_REQUIRE_GPU=1 but no CUDA driver/NVRTC found"
            );
            eprintln!("SKIP: no CUDA driver found; GPU test not run");
            return None;
        }
        Some(GpuRouter::new(0).expect("GpuRouter::new(0) failed"))
    });
    router.as_ref()
}

/// Deterministic pseudo-random logits in [-4, 4) (xorshift; no rand dep).
pub fn logits(num_tokens: usize, num_experts: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..num_tokens * num_experts)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
        })
        .collect()
}
