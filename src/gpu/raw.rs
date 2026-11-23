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
