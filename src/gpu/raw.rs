//! The only module in the crate allowed to use `unsafe`.
//!
//! Everything here wraps a cudarc call that the compiler cannot check for us:
//! probing for driver libraries, kernel launches (argument lists are untyped
//! `void*` arrays on the CUDA side), and marking our `#[repr(C)]` structs as
//! plain-old-data that can cross the host/device boundary.
#![allow(unsafe_code)]

/// True if both the CUDA driver (`libcuda`) and NVRTC can be loaded.
///
/// cudarc's dynamic loader panics on first use when a library is missing, so
/// every GPU entry point checks this first and returns an error instead.
pub fn cuda_available() -> bool {
    // SAFETY: these functions only attempt to `dlopen` candidate library names
    // and immediately drop the handle; they touch no CUDA state and have no
    // preconditions beyond being called from a normal thread.
    unsafe { cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present() }
}

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::router::Route;
use crate::telemetry::TelemetryRecord;

// SAFETY: `TelemetryRecord` is `#[repr(C)]` { u32, u16, u16, f32 } = 12 bytes,
// no padding (size asserted in telemetry.rs), plain data, same layout as the
// CUDA `TelemetryRecord`.
unsafe impl DeviceRepr for TelemetryRecord {}
// SAFETY: all-zero bits are a valid `TelemetryRecord`.
unsafe impl ValidAsZeroBits for TelemetryRecord {}

// SAFETY: `Route` is `#[repr(C)]` { u32, f32 }: no padding, no pointers, no
// drop glue, identical layout to the CUDA `RouteEntry` struct.
unsafe impl DeviceRepr for Route {}
// SAFETY: all-zero bits are a valid `Route` (expert 0, gate 0.0).
unsafe impl ValidAsZeroBits for Route {}

/// Threads per block for the one-thread-per-token kernels. Each thread holds
/// two MAX_K arrays, so a modest block size keeps register pressure sane.
const BLOCK: u32 = 256;

/// Grid covering `num_tokens` threads, one token each.
fn per_token_config(num_tokens: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (num_tokens.div_ceil(BLOCK), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
}

/// Launches `moe_route` on `stream`.
///
/// Preconditions (checked by the safe caller in `gpu::mod`, asserted here):
/// `logits.len() == num_tokens * num_experts`, `out.len() == num_tokens * k`,
/// `1 <= k <= MAX_K`, `num_tokens > 0`.
pub(super) fn launch_route(
    stream: &CudaStream,
    func: &CudaFunction,
    logits: &CudaSlice<f32>,
    num_tokens: u32,
    num_experts: u32,
    k: u32,
    mode: i32,
    out: &mut CudaSlice<Route>,
) -> Result<(), cudarc::driver::DriverError> {
    assert_eq!(logits.len(), num_tokens as usize * num_experts as usize);
    assert_eq!(out.len(), num_tokens as usize * k as usize);
    assert!(k >= 1 && k as usize <= super::kernels::MAX_K && num_tokens > 0);
    let mut launch = stream.launch_builder(func);
    launch.arg(logits).arg(&num_tokens).arg(&num_experts).arg(&k).arg(&mode).arg(out);
    // SAFETY: argument order and types match the kernel signature
    // (const float*, uint, uint, uint, int, RouteEntry*). Thread t reads
    // logits[t*E .. t*E+E) and writes out[t*k .. t*k+k) only for t < num_tokens,
    // which the length asserts above keep in bounds; k <= MAX_K keeps the
    // per-thread arrays in bounds.
    unsafe { launch.launch(per_token_config(num_tokens)) }?;
    Ok(())
}

/// Device buffers and scalars for one `moe_route_dispatch_log` launch.
pub(super) struct DispatchLaunch<'a> {
    pub logits: &'a CudaSlice<f32>,
    pub num_tokens: u32,
    pub num_experts: u32,
    pub k: u32,
    pub mode: i32,
    pub layer_id: u32,
    pub token_offset: u32,
    pub routes: &'a mut CudaSlice<Route>,
    pub expert_counts: &'a mut CudaSlice<u32>,
    pub gathered: &'a mut CudaSlice<u32>,
    pub log: &'a mut CudaSlice<TelemetryRecord>,
    pub log_counters: &'a mut CudaSlice<u64>,
}

/// Launches `moe_route_dispatch_log` on `stream`.
///
/// The length checks below are the whole safety story for the kernel's
/// writes; the safe caller in `gpu::mod` sizes the buffers, and these
/// asserts make a sizing bug panic instead of corrupting device memory.
pub(super) fn launch_route_dispatch_log(
    stream: &CudaStream,
    func: &CudaFunction,
    p: DispatchLaunch<'_>,
) -> Result<(), cudarc::driver::DriverError> {
    let (n, e, k) = (p.num_tokens as usize, p.num_experts as usize, p.k as usize);
    assert!(n > 0 && k >= 1 && k <= super::kernels::MAX_K && k <= e);
    assert_eq!(p.logits.len(), n * e);
    assert_eq!(p.routes.len(), n * k);
    assert_eq!(p.expert_counts.len(), e);
    assert_eq!(p.gathered.len(), n * e);
    assert_eq!(p.log_counters.len(), 2);
    assert!(e <= u16::MAX as usize + 1 && p.layer_id <= u16::MAX as u32);
    assert!(p.token_offset as u64 + n as u64 <= u32::MAX as u64 + 1);
    let capacity = p.log.len() as u64;

    let mut launch = stream.launch_builder(func);
    launch
        .arg(p.logits)
        .arg(&p.num_tokens)
        .arg(&p.num_experts)
        .arg(&p.k)
        .arg(&p.mode)
        .arg(&p.layer_id)
        .arg(&p.token_offset)
        .arg(p.routes)
        .arg(p.expert_counts)
        .arg(p.gathered)
        .arg(p.log)
        .arg(&capacity)
        .arg(p.log_counters);
    // SAFETY: argument order and types match the kernel signature
    // (const float*, uint x6 [one is int mode], RouteEntry*, uint*, uint*,
    // TelemetryRecord*, unsigned long long, unsigned long long*). With the
    // asserts above, for every thread t < n:
    // * logits reads [t*E, t*E+E) and routes writes [t*k, t*k+k): in bounds;
    // * expert ids are < E, so expert_counts[e] is in bounds, and the slot
    //   returned is < n (counts were zeroed and each token hits e at most
    //   once), so gathered[e*n + slot] < E*n is in bounds;
    // * log is written only at idx < capacity == log.len();
    // * log_counters has exactly the 2 elements the kernel touches.
    // All flat indices fit in u32 (checked by the caller), so none wrap.
    unsafe { launch.launch(per_token_config(p.num_tokens)) }?;
    Ok(())
}
