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

use cudarc::driver::sys::{
    CUdeviceptr, CUpointer_attribute, cuMemGetAddressRange_v2, cuPointerGetAttribute,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};

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
    LaunchConfig {
        grid_dim: (num_tokens.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launches `moe_route` on `stream`.
///
/// Preconditions (checked by the safe caller in `gpu::mod`, asserted here):
/// `logits.len() == num_tokens * num_experts`, `out.len() == num_tokens * k`,
/// `1 <= k <= MAX_K`, `num_tokens > 0`.
#[allow(clippy::too_many_arguments)] // mirrors the kernel signature
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
    launch
        .arg(logits)
        .arg(&num_tokens)
        .arg(&num_experts)
        .arg(&k)
        .arg(&mode)
        .arg(out);
    // SAFETY: argument order and types match the kernel signature
    // (const float*, uint, uint, uint, int, RouteEntry*). Thread t reads
    // logits[t*E .. t*E+E) and writes out[t*k .. t*k+k) only for t < num_tokens,
    // which the length asserts above keep in bounds; k <= MAX_K keeps the
    // per-thread arrays in bounds.
    unsafe { launch.launch(per_token_config(num_tokens)) }?;
    Ok(())
}

/// A device pointer owned by another library (e.g. a torch CUDA tensor's
/// `data_ptr()`), checked against the driver to cover `len` f32s inside one
/// live allocation on our device. Only constructible via [`borrow_foreign_f32`].
#[derive(Debug, Clone, Copy)]
pub struct ForeignF32 {
    ptr: CUdeviceptr,
    len: usize,
}

impl ForeignF32 {
    /// Number of f32 elements covered.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Why a foreign pointer was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForeignPtrError {
    Null,
    Misaligned,
    /// The driver does not know this address as device memory.
    NotDeviceMemory,
    /// The allocation is on a different GPU than the router.
    WrongDevice {
        expected: usize,
        found: i32,
    },
    /// `[ptr, ptr + len*4)` runs past the end of the allocation.
    OutOfRange {
        needed: usize,
        available: usize,
    },
}

/// Validates `ptr` as `len` f32s of device memory usable from `ctx`.
///
/// This is what lets the GPU path take a raw integer from Python without
/// trusting it: the kernel will only ever read inside an allocation the
/// driver confirms is live, on our device, and large enough.
///
/// Granularity is the *driver allocation*, not the tensor: caching
/// allocators (torch) carve many tensors out of one segment, so a wrong
/// `len` can still read a neighbouring tensor's bytes (wrong telemetry) -
/// but never unmapped or freed memory.
pub(super) fn borrow_foreign_f32(
    ctx: &CudaContext,
    ptr: u64,
    len: usize,
) -> Result<Result<ForeignF32, ForeignPtrError>, cudarc::driver::DriverError> {
    if ptr == 0 {
        return Ok(Err(ForeignPtrError::Null));
    }
    if !ptr.is_multiple_of(std::mem::align_of::<f32>() as u64) {
        return Ok(Err(ForeignPtrError::Misaligned));
    }
    ctx.bind_to_thread()?;
    let (mut base, mut size): (CUdeviceptr, usize) = (0, 0);
    // SAFETY: both out-pointers refer to live, correctly typed locals; the
    // driver only reads `ptr` as an address (it is never dereferenced on
    // the host) and returns an error code for unknown addresses.
    let rc = unsafe { cuMemGetAddressRange_v2(&mut base, &mut size, ptr) };
    if rc.result().is_err() {
        return Ok(Err(ForeignPtrError::NotDeviceMemory));
    }
    let mut ordinal: i32 = -1;
    // SAFETY: DEVICE_ORDINAL writes a single `int`, and `ordinal` is an i32 local.
    let rc = unsafe {
        cuPointerGetAttribute(
            (&mut ordinal as *mut i32).cast(),
            CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            ptr,
        )
    };
    rc.result()?;
    if ordinal != ctx.ordinal() as i32 {
        return Ok(Err(ForeignPtrError::WrongDevice {
            expected: ctx.ordinal(),
            found: ordinal,
        }));
    }
    let needed = len.saturating_mul(4);
    let available = (base as usize + size).saturating_sub(ptr as usize);
    if needed > available {
        return Ok(Err(ForeignPtrError::OutOfRange { needed, available }));
    }
    Ok(Ok(ForeignF32 { ptr, len }))
}

/// Where a kernel reads its logits from.
pub(super) enum LogitsArg<'a> {
    /// A buffer we allocated and filled from host memory.
    Owned(&'a CudaSlice<f32>),
    /// Another library's device memory, validated by [`borrow_foreign_f32`].
    Foreign(ForeignF32),
}

impl LogitsArg<'_> {
    fn len(&self) -> usize {
        match self {
            LogitsArg::Owned(s) => s.len(),
            LogitsArg::Foreign(f) => f.len,
        }
    }
}

/// Device buffers and scalars for one `moe_route_dispatch_log` launch.
pub(super) struct DispatchLaunch<'a> {
    pub logits: LogitsArg<'a>,
    pub num_tokens: u32,
    pub num_experts: u32,
    pub k: u32,
    pub mode: i32,
    pub layer_id: u32,
    pub token_offset: u32,
    /// `None` = telemetry-only: the kernel skips routes/dispatch entirely.
    pub dispatch: Option<DispatchOutputs<'a>>,
    pub log: &'a mut CudaSlice<TelemetryRecord>,
    pub log_counters: &'a mut CudaSlice<u64>,
}

/// Per-call outputs of the dispatch half of the kernel.
pub(super) struct DispatchOutputs<'a> {
    pub routes: &'a mut CudaSlice<Route>,
    pub expert_counts: &'a mut CudaSlice<u32>,
    pub gathered: &'a mut CudaSlice<u32>,
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
    assert!(n > 0 && (1..=super::kernels::MAX_K).contains(&k) && k <= e);
    assert_eq!(p.logits.len(), n * e);
    if let Some(d) = &p.dispatch {
        assert_eq!(d.routes.len(), n * k);
        assert_eq!(d.expert_counts.len(), e);
        assert_eq!(d.gathered.len(), n * e);
    }
    assert_eq!(p.log_counters.len(), 2);
    assert!(e <= u16::MAX as usize + 1 && p.layer_id <= u16::MAX as u32);
    assert!(p.token_offset as u64 + n as u64 <= u32::MAX as u64 + 1);
    let capacity = p.log.len() as u64;

    let mut launch = stream.launch_builder(func);
    let foreign_ptr: CUdeviceptr;
    match &p.logits {
        LogitsArg::Owned(slice) => launch.arg(*slice),
        LogitsArg::Foreign(f) => {
            foreign_ptr = f.ptr;
            // A `const float*` kernel parameter is just the 8-byte device address.
            launch.arg(&foreign_ptr)
        }
    };
    launch
        .arg(&p.num_tokens)
        .arg(&p.num_experts)
        .arg(&p.k)
        .arg(&p.mode)
        .arg(&p.layer_id)
        .arg(&p.token_offset);
    // Null device pointers for the unused dispatch outputs in telemetry-only mode.
    let null: CUdeviceptr = 0;
    let write_dispatch: i32 = p.dispatch.is_some() as i32;
    match p.dispatch {
        Some(d) => launch.arg(d.routes).arg(d.expert_counts).arg(d.gathered),
        None => launch.arg(&null).arg(&null).arg(&null),
    };
    launch
        .arg(p.log)
        .arg(&capacity)
        .arg(p.log_counters)
        .arg(&write_dispatch);
    // SAFETY: argument order and types match the kernel signature
    // (const float*, uint x6 [one is int mode], RouteEntry*, uint*, uint*,
    // TelemetryRecord*, unsigned long long, unsigned long long*, int). When
    // `dispatch` is None the three output pointers are null and
    // write_dispatch == 0, so the kernel never dereferences them. With the
    // asserts above, for every thread t < n:
    // * logits reads [t*E, t*E+E) and routes writes [t*k, t*k+k): in bounds.
    //   For foreign logits, borrow_foreign_f32 proved the n*E floats lie in
    //   one live allocation on this device. It stays live for the kernel:
    //   the Python caller holds the tensor across this call, and torch only
    //   reuses freed blocks in stream order after our launch on the legacy
    //   default stream (cudaFree itself synchronizes the device);
    // * expert ids are < E, so expert_counts[e] is in bounds, and the slot
    //   returned is < n (counts were zeroed and each token hits e at most
    //   once), so gathered[e*n + slot] < E*n is in bounds;
    // * log is written only at idx < capacity == log.len();
    // * log_counters has exactly the 2 elements the kernel touches.
    // All flat indices fit in u32 (checked by the caller), so none wrap.
    unsafe { launch.launch(per_token_config(p.num_tokens)) }?;
    Ok(())
}
