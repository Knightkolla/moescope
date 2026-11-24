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

use std::sync::Arc;

use cudarc::driver::sys::{
    CUcontext, CUdeviceptr, CUpointer_attribute, CUstream, cuMemGetAddressRange_v2,
    cuPointerGetAttribute, cuStreamGetCtx,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};

use crate::router::{GateMode, Gating, Route};
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

/// Element type of router logits in device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsDtype {
    F32,
    F16,
    BF16,
}

impl LogitsDtype {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            LogitsDtype::F32 => 4,
            LogitsDtype::F16 | LogitsDtype::BF16 => 2,
        }
    }

    /// Kernel encoding (`GateParams.logits_dtype`).
    fn code(self) -> i32 {
        match self {
            LogitsDtype::F32 => 0,
            LogitsDtype::F16 => 1,
            LogitsDtype::BF16 => 2,
        }
    }
}

/// Kernel-side parameters, passed by value. Mirrors `GateParams` in the
/// CUDA source: { int, uint, uint, int, float, int, const float*,
/// const unsigned long long*, int, int }, 48 bytes, pointers 8-byte aligned
/// at 24, 32.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GateParams {
    mode: i32,
    num_groups: u32,
    top_groups: u32,
    normalize: i32,
    scale: f32,
    logits_dtype: i32,
    bias: CUdeviceptr,
    token_base: CUdeviceptr,
    round_biased: i32,
    pad: i32,
}
const _: () = assert!(std::mem::size_of::<GateParams>() == 48);

// SAFETY: `GateParams` is `#[repr(C)]` plain data (integers, a float and a
// device address) with the same layout as the CUDA struct (size asserted).
unsafe impl DeviceRepr for GateParams {}

/// Gating for one launch: the configuration plus an optional per-expert
/// selection bias already on the device (validated by the caller to be
/// `num_experts` long and only used in sigmoid mode).
#[derive(Clone, Copy)]
pub(super) struct GateArg<'a> {
    pub gating: Gating,
    pub bias: Option<&'a CudaSlice<f32>>,
    /// Round score + bias to the logits' half precision before ranking.
    pub round_biased: bool,
}

impl GateArg<'_> {
    /// Asserts the invariants the kernel relies on.
    fn check(&self, num_experts: usize) {
        if let Some(b) = self.bias {
            assert_eq!(b.len(), num_experts, "bias length");
            assert_eq!(self.gating.mode, GateMode::Sigmoid);
        }
        if let Some(g) = self.gating.groups {
            assert!(g.num_groups >= 1 && g.num_groups <= crate::router::MAX_GROUPS);
            assert!(num_experts.is_multiple_of(g.num_groups));
            assert!(g.top_groups >= 1 && g.top_groups <= g.num_groups);
        }
    }
}

/// Pushes `gate` as the kernel's `GateParams` argument. The returned guard
/// (cudarc's stream-ordering hook for the bias buffer) must outlive the
/// launch, so the caller keeps it until after `launch`.
type Guard<'a> = Option<cudarc::driver::SyncOnDrop<'a>>;

/// Device address of an optional buffer (0 for None) plus cudarc's
/// stream-ordering guard, which must outlive the launch that uses it.
fn ptr_of<'a, T>(
    buf: Option<&'a CudaSlice<T>>,
    stream: &'a CudaStream,
) -> (CUdeviceptr, Guard<'a>) {
    match buf {
        Some(b) => {
            let (ptr, guard) = b.device_ptr(stream);
            (ptr, Some(guard))
        }
        None => (0, None),
    }
}

fn push_gate<'a>(
    stream: &'a CudaStream,
    gate: GateArg<'a>,
    logits_dtype: LogitsDtype,
    token_base: Option<&'a CudaSlice<u64>>,
) -> (GateParams, [Guard<'a>; 2]) {
    let (bias, bias_guard) = ptr_of(gate.bias, stream);
    let (token_base, token_guard) = ptr_of(token_base, stream);
    let g = gate.gating;
    let (num_groups, top_groups) = g
        .groups
        .map_or((1, 1), |x| (x.num_groups as u32, x.top_groups as u32));
    let params = GateParams {
        mode: match g.mode {
            GateMode::SoftmaxThenTopK => 0,
            GateMode::TopKThenSoftmax => 1,
            GateMode::Sigmoid => 2,
        },
        num_groups,
        top_groups,
        normalize: g.normalize as i32,
        scale: g.scale,
        logits_dtype: logits_dtype.code(),
        bias,
        token_base,
        round_biased: gate.round_biased as i32,
        pad: 0,
    };
    (params, [bias_guard, token_guard])
}

/// Threads per block. A block holds `BLOCK / lanes` tokens; every lane keeps
/// two MAX_K arrays, so a modest block size keeps register pressure sane.
const BLOCK: u32 = 256;

/// Grid covering `num_tokens` tokens with `lanes` threads each. The total is
/// rounded up to whole blocks, so every warp is full (the kernels' shuffles
/// need all 32 lanes) and threads past the last token sit out.
fn per_token_config(num_tokens: u32, lanes: u32) -> LaunchConfig {
    let threads = num_tokens as u64 * lanes as u64;
    let blocks = threads.div_ceil(BLOCK as u64);
    assert!(blocks <= i32::MAX as u64, "grid too large");
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
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
    gate: GateArg<'_>,
    out: &mut CudaSlice<Route>,
) -> Result<(), cudarc::driver::DriverError> {
    assert_eq!(logits.len(), num_tokens as usize * num_experts as usize);
    assert_eq!(out.len(), num_tokens as usize * k as usize);
    assert!(k >= 1 && k as usize <= super::kernels::MAX_K && num_tokens > 0);
    gate.check(num_experts as usize);
    let lanes = super::kernels::lanes_per_token(num_experts);
    let (params, _guards) = push_gate(stream, gate, LogitsDtype::F32, None);
    let mut launch = stream.launch_builder(func);
    launch
        .arg(logits)
        .arg(&num_tokens)
        .arg(&num_experts)
        .arg(&k)
        .arg(&params)
        .arg(out)
        .arg(&lanes);
    // SAFETY: argument order and types match the kernel signature
    // (const float*, uint, uint, uint, GateParams, RouteEntry*, uint).
    // GateParams.bias is null or a live device buffer of exactly E floats
    // (gate.check), read only at indices < E. The lanes of
    // token t read logits[t*E .. t*E+E) and write out[t*k .. t*k+k) only for
    // t < num_tokens, which the length asserts above keep in bounds; k <= MAX_K
    // keeps the per-lane arrays in bounds; lanes is a power of two <= 32 that
    // divides the block, as the kernel's shuffles require.
    unsafe { launch.launch(per_token_config(num_tokens, lanes)) }?;
    Ok(())
}

/// A device pointer owned by another library (e.g. a torch CUDA tensor's
/// `data_ptr()`), checked against the driver to cover `len` f32s inside one
/// live allocation on our device. Only constructible via [`borrow_foreign`].
#[derive(Debug, Clone, Copy)]
pub struct ForeignLogits {
    ptr: CUdeviceptr,
    len: usize,
    dtype: LogitsDtype,
}

impl ForeignLogits {
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
    /// `[ptr, ptr + len * element size)` runs past the end of the allocation.
    OutOfRange {
        needed: usize,
        available: usize,
    },
    /// A stream handle that is not a stream of this GPU's context.
    ForeignStream,
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
pub(super) fn borrow_foreign(
    ctx: &CudaContext,
    ptr: u64,
    len: usize,
    dtype: LogitsDtype,
) -> Result<Result<ForeignLogits, ForeignPtrError>, cudarc::driver::DriverError> {
    if ptr == 0 {
        return Ok(Err(ForeignPtrError::Null));
    }
    if !ptr.is_multiple_of(dtype.size() as u64) {
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
    let needed = len.saturating_mul(dtype.size());
    let available = (base as usize + size).saturating_sub(ptr as usize);
    if needed > available {
        return Ok(Err(ForeignPtrError::OutOfRange { needed, available }));
    }
    Ok(Ok(ForeignLogits { ptr, len, dtype }))
}

/// Where a kernel reads its logits from.
pub(super) enum LogitsArg<'a> {
    /// A buffer we allocated and filled from host memory.
    Owned(&'a CudaSlice<f32>),
    /// Another library's device memory, validated by [`borrow_foreign`].
    Foreign(ForeignLogits),
}

impl LogitsArg<'_> {
    fn len(&self) -> usize {
        match self {
            LogitsArg::Owned(s) => s.len(),
            LogitsArg::Foreign(f) => f.len,
        }
    }

    pub(super) fn dtype(&self) -> LogitsDtype {
        match self {
            LogitsArg::Owned(_) => LogitsDtype::F32,
            LogitsArg::Foreign(f) => f.dtype,
        }
    }
}

/// Wraps another library's CUDA stream (e.g. torch's current stream,
/// `torch.cuda.current_stream().cuda_stream`) so our kernels are ordered
/// with the model's work without any host synchronization, and so they can
/// be recorded into a CUDA graph being captured on that stream.
///
/// `handle == 0` (the legacy default stream) returns `None`: use the
/// router's own default stream, which is that same stream.
pub(super) fn borrow_stream(
    ctx: &Arc<CudaContext>,
    handle: u64,
) -> Result<Result<Option<Arc<CudaStream>>, ForeignPtrError>, cudarc::driver::DriverError> {
    if handle == 0 {
        return Ok(Ok(None));
    }
    ctx.bind_to_thread()?;
    let stream = handle as usize as CUstream;
    let mut owner: CUcontext = std::ptr::null_mut();
    // SAFETY: `owner` is a live local of the out-parameter's type; the driver
    // only inspects `stream` and returns an error for an invalid handle.
    let rc = unsafe { cuStreamGetCtx(stream, &mut owner) };
    if rc.result().is_err() || owner != ctx.cu_ctx() {
        return Ok(Err(ForeignPtrError::ForeignStream));
    }
    // SAFETY: the driver just confirmed `stream` is a valid stream of this
    // context. The caller (Python, holding the torch stream) keeps it alive
    // for the duration of the call; the wrapper is not owning, so dropping
    // it never destroys the stream, and work already enqueued on a stream
    // completes even if its owner destroys it later. Our buffers are used on
    // it only through stream-ordered launches; `TelemetryBuffer::drain`
    // synchronizes the whole context before reading them.
    Ok(Ok(Some(unsafe { ctx.wrap_borrowed_stream(stream) })))
}

/// Launches `moe_advance_tokens`: `*token_base += n` on `stream`.
pub(super) fn launch_advance_tokens(
    stream: &CudaStream,
    func: &CudaFunction,
    token_base: &mut CudaSlice<u64>,
    n: u64,
) -> Result<(), cudarc::driver::DriverError> {
    assert_eq!(token_base.len(), 1);
    let mut launch = stream.launch_builder(func);
    launch.arg(token_base).arg(&n);
    // SAFETY: matches `moe_advance_tokens(unsigned long long*, unsigned long
    // long)`; one thread writes element 0 of a 1-element buffer.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

/// Device buffers and scalars for one `moe_route_dispatch_log` launch.
pub(super) struct DispatchLaunch<'a> {
    pub logits: LogitsArg<'a>,
    pub num_tokens: u32,
    pub num_experts: u32,
    pub k: u32,
    pub gate: GateArg<'a>,
    /// Device token counter added to token ids (CUDA-graph-safe), or None.
    pub token_base: Option<&'a CudaSlice<u64>>,
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
    let lanes = super::kernels::lanes_per_token(p.num_experts);
    p.gate.check(e);
    if let Some(tb) = p.token_base {
        assert_eq!(tb.len(), 1);
    }
    let (params, _guards) = push_gate(stream, p.gate, p.logits.dtype(), p.token_base);

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
        .arg(&params)
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
        .arg(&write_dispatch)
        .arg(&lanes);
    // SAFETY: argument order and types match the kernel signature
    // (const float*, uint x3, GateParams, uint x2, RouteEntry*, uint*, uint*,
    // TelemetryRecord*, unsigned long long, unsigned long long*, int, uint
    // lanes: a power of two <= 32 dividing the block, so groups never
    // straddle a warp and every warp is full for the shuffles). When
    // `dispatch` is None the three output pointers are null and
    // write_dispatch == 0, so the kernel never dereferences them. With the
    // asserts above, for every token t < n (any of its lanes):
    // * logits reads [t*E, t*E+E) and routes writes [t*k, t*k+k): in bounds.
    //   For foreign logits, borrow_foreign_f32 proved the n*E floats lie in
    //   one live allocation on this device. It stays live for the kernel:
    //   the Python caller holds the tensor across this call, and torch only
    //   reuses freed blocks in stream order after our launch on the legacy
    //   default stream (cudaFree itself synchronizes the device);
    // * expert ids are < E, so expert_counts[e] is in bounds, and the slot
    //   returned is < n (counts were zeroed and each token hits e at most
    //   once), so gathered[e*n + slot] < E*n is in bounds;
    // * GateParams.bias is null or a live device buffer of exactly E floats
    //   (gate.check), read only at expert indices < E; token_base is null or
    //   a live 1-element u64 buffer (asserted), only read;
    // * logits are read as GateParams.logits_dtype, the dtype the foreign
    //   pointer was validated for (n*E elements of that size);
    // * log is written only at idx < capacity == log.len();
    // * log_counters has exactly the 2 elements the kernel touches.
    // All flat indices fit in u32 (checked by the caller), so none wrap.
    unsafe { launch.launch(per_token_config(p.num_tokens, lanes)) }?;
    Ok(())
}
