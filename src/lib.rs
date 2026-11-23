//! moescope: Mixture-of-Experts routing telemetry.
//!
//! * [`router`]: CPU reference router (top-k + gate weights).
//! * [`dispatcher`]: groups routed tokens per expert and builds the load histogram.
//! * [`telemetry`]: the fixed-size record format and a host-side log.
//! * `gpu` (feature `cuda`): the same pipeline as NVRTC-compiled CUDA kernels,
//!   logging every routing decision into device memory.
//!
//! `unsafe` is denied crate-wide and allowed only in `gpu::raw`.
#![deny(unsafe_code)]

pub mod dispatcher;
pub mod router;
pub mod telemetry;

#[cfg(feature = "cuda")]
pub mod gpu;
