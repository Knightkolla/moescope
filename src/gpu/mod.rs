//! GPU path: CUDA C kernels compiled at runtime with NVRTC, launched via cudarc.

pub mod kernels;
mod raw;

pub use raw::{ForeignF32, ForeignPtrError, cuda_available};

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};

use crate::dispatcher::DispatchPlan;
use crate::router::{GateMode, Route, RouteError, check_shape, validate_batch};
use crate::telemetry::{Drained, TelemetryError, TelemetryRecord, check_token_range};

/// Anything that can go wrong on the GPU path.
#[derive(Debug)]
pub enum GpuError {
    /// `libcuda` or NVRTC could not be loaded (no NVIDIA driver / toolkit).
    CudaUnavailable,
    /// CUDA driver call failed (no GPU, out of memory, bad launch, ...).
    Driver(DriverError),
    /// NVRTC rejected the kernel source.
    Compile(CompileError),
    /// The routing request itself is invalid (same checks as the CPU router).
    Route(RouteError),
    /// The request is valid for the CPU router but exceeds a GPU kernel limit.
    Limit(String),
    /// The batch can't be expressed as telemetry records (id overflow).
    Telemetry(TelemetryError),
    /// A device pointer from another library failed validation.
    ForeignPtr(ForeignPtrError),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::CudaUnavailable => write!(f, "CUDA driver or NVRTC library not found"),
            GpuError::Driver(e) => write!(f, "CUDA driver error: {e:?}"),
            GpuError::Compile(e) => write!(f, "NVRTC compile error: {e:?}"),
            GpuError::Route(e) => write!(f, "{e}"),
            GpuError::Limit(msg) => write!(f, "GPU kernel limit: {msg}"),
            GpuError::Telemetry(e) => write!(f, "{e}"),
            GpuError::ForeignPtr(e) => write!(f, "invalid device pointer: {e:?}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::Driver(e)
    }
}

impl From<RouteError> for GpuError {
    fn from(e: RouteError) -> Self {
        GpuError::Route(e)
    }
}

impl From<TelemetryError> for GpuError {
    fn from(e: TelemetryError) -> Self {
        GpuError::Telemetry(e)
    }
}

impl From<CompileError> for GpuError {
    fn from(e: CompileError) -> Self {
        GpuError::Compile(e)
    }
}

/// Opens the primary CUDA context on GPU `ordinal`, or reports
/// [`GpuError::CudaUnavailable`] instead of panicking when there is no driver.
pub fn open_context(ordinal: usize) -> Result<Arc<CudaContext>, GpuError> {
    if !cuda_available() {
        return Err(GpuError::CudaUnavailable);
    }
    Ok(CudaContext::new(ordinal)?)
}

/// NVRTC `--gpu-architecture` flag for the device behind `ctx`, e.g.
/// `compute_75` on a T4. Targeting the real device matters because newer
/// NVRTC releases dropped their old default architectures.
pub fn nvrtc_arch_flag(ctx: &CudaContext) -> Result<String, GpuError> {
    let (major, minor) = ctx.compute_capability()?;
    Ok(format!("--gpu-architecture=compute_{major}{minor}"))
}

/// Compiles CUDA C source to PTX for this device and loads it as a module.
pub fn compile_module(ctx: &Arc<CudaContext>, src: &str) -> Result<Arc<CudaModule>, GpuError> {
    let opts = CompileOptions {
        options: vec![nvrtc_arch_flag(ctx)?],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(src, opts)?;
    Ok(ctx.load_module(ptx)?)
}

/// `alloc_zeros`, but never zero bytes (`cuMemAlloc(0)` is an error).
fn alloc_zeros_nonempty<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>, GpuError> {
    Ok(stream.alloc_zeros::<T>(len.max(1))?)
}

/// Copies the first `len` elements of a device buffer to host.
fn copy_prefix<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<T>,
    len: usize,
) -> Result<Vec<T>, GpuError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    Ok(stream.clone_dtoh(&buf.slice(0..len))?)
}

/// Kernel argument encoding of [`GateMode`] (must match `compute_gates`).
fn mode_code(mode: GateMode) -> i32 {
    match mode {
        GateMode::SoftmaxThenTopK => 0,
        GateMode::TopKThenSoftmax => 1,
    }
}

/// Checked conversion for kernel `unsigned int` arguments.
fn to_u32(value: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::Limit(format!("{what} = {value} exceeds u32")))
}

/// Shape of one validated routing batch, already converted to kernel types.
#[derive(Debug, Clone, Copy)]
struct BatchShape {
    num_tokens: u32,
    num_experts: u32,
    k: u32,
}

/// Runs the CPU router's validation plus the kernel-specific limits
/// (`k <= MAX_K`, all flat indices fit in `u32`).
fn validate_for_gpu(logits: &[f32], num_experts: usize, k: usize) -> Result<BatchShape, GpuError> {
    let num_tokens = validate_batch(logits, num_experts, k)?;
    validate_shape_for_gpu(num_tokens, num_experts, k)
}

/// Shape-only validation, for logits that live on the device.
fn validate_shape_for_gpu(
    num_tokens: usize,
    num_experts: usize,
    k: usize,
) -> Result<BatchShape, GpuError> {
    check_shape(num_experts, k)?;
    if k > kernels::MAX_K {
        return Err(GpuError::Limit(format!(
            "k = {k} exceeds MAX_K = {}",
            kernels::MAX_K
        )));
    }
    to_u32(num_tokens * k, "num_tokens * k")?;
    to_u32(num_tokens * num_experts, "num_tokens * num_experts")?;
    Ok(BatchShape {
        num_tokens: to_u32(num_tokens, "num_tokens")?,
        num_experts: to_u32(num_experts, "num_experts")?,
        k: k as u32,
    })
}

/// GPU MoE router: owns the CUDA context, a stream, and the compiled kernels.
///
/// Create once per process (NVRTC compilation takes ~100 ms) and reuse.
pub struct GpuRouter {
    stream: Arc<CudaStream>,
    route_fn: CudaFunction,
    dispatch_log_fn: CudaFunction,
}

impl GpuRouter {
    /// Opens GPU `ordinal` and compiles the routing kernels for it.
    pub fn new(ordinal: usize) -> Result<Self, GpuError> {
        let ctx = open_context(ordinal)?;
        let module = compile_module(&ctx, &kernels::source())?;
        Ok(GpuRouter {
            route_fn: module.load_function(kernels::ROUTE_KERNEL)?,
            dispatch_log_fn: module.load_function(kernels::DISPATCH_LOG_KERNEL)?,
            stream: ctx.default_stream(),
        })
    }

    /// Routes a `[num_tokens, num_experts]` batch on the GPU and copies the
    /// result back. Same contract and `[token * k + j]` layout as
    /// [`crate::router::route_batch`]; that function is its test oracle.
    pub fn route(
        &self,
        logits: &[f32],
        num_experts: usize,
        k: usize,
        mode: GateMode,
    ) -> Result<Vec<Route>, GpuError> {
        let shape = validate_for_gpu(logits, num_experts, k)?;
        if shape.num_tokens == 0 {
            return Ok(Vec::new());
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        let mut out = self
            .stream
            .alloc_zeros::<Route>(shape.num_tokens as usize * k)?;
        raw::launch_route(
            &self.stream,
            &self.route_fn,
            &logits_dev,
            shape.num_tokens,
            shape.num_experts,
            shape.k,
            mode_code(mode),
            &mut out,
        )?;
        Ok(self.stream.clone_dtoh(&out)?)
    }

    /// Routes, dispatches and logs one MoE layer's batch in a single kernel.
    ///
    /// Every (token, expert) decision is appended to `log` *in device
    /// memory*; nothing is copied back here. Call [`TelemetryBuffer::drain`]
    /// once at the end of the logging window. The returned [`DeviceDispatch`]
    /// keeps the routes, the per-expert token groups, and the load histogram
    /// on the device; copy them back only if you need them.
    ///
    /// `layer_id` tags the records; `token_offset` is added to each token
    /// index so successive batches get distinct `token_id`s. Logits are
    /// validated exactly like the CPU router (shape, finite).
    #[allow(clippy::too_many_arguments)]
    pub fn route_dispatch_log(
        &self,
        logits: &[f32],
        num_experts: usize,
        k: usize,
        mode: GateMode,
        layer_id: u16,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<DeviceDispatch, GpuError> {
        let shape = validate_for_gpu(logits, num_experts, k)?;
        if shape.num_tokens == 0 {
            return self.empty_dispatch(num_experts);
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        self.launch_dispatch(
            raw::LogitsArg::Owned(&logits_dev),
            shape,
            mode,
            layer_id,
            token_offset,
            log,
        )
    }

    /// [`route_dispatch_log`](Self::route_dispatch_log) reading logits
    /// straight from another library's device memory (zero-copy), e.g. a
    /// contiguous float32 torch CUDA tensor's `data_ptr()`.
    ///
    /// The pointer is validated with the driver (live allocation on this
    /// GPU, large enough, aligned), so a bad integer is an error, not a
    /// wild read. (Per driver allocation: with a pooling allocator an
    /// oversized `num_tokens` can still read a neighbouring tensor.) Requirements on the caller for *correct* results: the
    /// logits must be complete in stream order before this launch (true when
    /// produced on the legacy default stream, which torch uses by default)
    /// and not overwritten until it finishes. Logits are not checked for
    /// NaN here (that would need a device read); the kernel ranks NaN as -inf.
    #[allow(clippy::too_many_arguments)]
    pub fn route_dispatch_log_device(
        &self,
        logits_ptr: u64,
        num_tokens: usize,
        num_experts: usize,
        k: usize,
        mode: GateMode,
        layer_id: u16,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<DeviceDispatch, GpuError> {
        let shape = validate_shape_for_gpu(num_tokens, num_experts, k)?;
        if num_tokens == 0 {
            return self.empty_dispatch(num_experts);
        }
        let foreign =
            raw::borrow_foreign_f32(self.stream.context(), logits_ptr, num_tokens * num_experts)?
                .map_err(GpuError::ForeignPtr)?;
        self.launch_dispatch(
            raw::LogitsArg::Foreign(foreign),
            shape,
            mode,
            layer_id,
            token_offset,
            log,
        )
    }

    /// Result for a zero-token batch: nothing to launch, empty histogram.
    fn empty_dispatch(&self, num_experts: usize) -> Result<DeviceDispatch, GpuError> {
        let stream = &self.stream;
        Ok(DeviceDispatch {
            stream: stream.clone(),
            num_experts,
            num_tokens: 0,
            k: 0,
            routes: alloc_zeros_nonempty::<Route>(stream, 0)?,
            expert_counts: stream.alloc_zeros::<u32>(num_experts)?,
            gathered: alloc_zeros_nonempty::<u32>(stream, 0)?,
        })
    }

    /// Telemetry only: like [`route_dispatch_log`](Self::route_dispatch_log)
    /// but nothing except the log is written. No per-call allocations, one
    /// kernel launch. Use this when you only want the routing record.
    #[allow(clippy::too_many_arguments)]
    pub fn log_only(
        &self,
        logits: &[f32],
        num_experts: usize,
        k: usize,
        mode: GateMode,
        layer_id: u16,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<(), GpuError> {
        let shape = validate_for_gpu(logits, num_experts, k)?;
        if shape.num_tokens == 0 {
            return Ok(());
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        let logits = raw::LogitsArg::Owned(&logits_dev);
        self.launch(logits, shape, mode, layer_id, token_offset, log, None)
    }

    /// Zero-copy [`log_only`](Self::log_only) for logits already on the GPU
    /// (same pointer validation and caller requirements as
    /// [`route_dispatch_log_device`](Self::route_dispatch_log_device)). This
    /// is what the Python `route_and_log` uses for CUDA tensors: one pointer
    /// check plus one kernel launch per call.
    #[allow(clippy::too_many_arguments)]
    pub fn log_only_device(
        &self,
        logits_ptr: u64,
        num_tokens: usize,
        num_experts: usize,
        k: usize,
        mode: GateMode,
        layer_id: u16,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<(), GpuError> {
        let shape = validate_shape_for_gpu(num_tokens, num_experts, k)?;
        if num_tokens == 0 {
            return Ok(());
        }
        let foreign =
            raw::borrow_foreign_f32(self.stream.context(), logits_ptr, num_tokens * num_experts)?
                .map_err(GpuError::ForeignPtr)?;
        let logits = raw::LogitsArg::Foreign(foreign);
        self.launch(logits, shape, mode, layer_id, token_offset, log, None)
    }

    /// Shared tail of the dispatch entry points: allocate the per-call
    /// outputs, then launch.
    fn launch_dispatch(
        &self,
        logits: raw::LogitsArg<'_>,
        shape: BatchShape,
        mode: GateMode,
        layer_id: u16,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<DeviceDispatch, GpuError> {
        let (n, e, k) = (
            shape.num_tokens as usize,
            shape.num_experts as usize,
            shape.k as usize,
        );
        let stream = &self.stream;
        let mut out = DeviceDispatch {
            stream: stream.clone(),
            num_experts: e,
            num_tokens: n,
            k,
            routes: alloc_zeros_nonempty::<Route>(stream, n * k)?,
            expert_counts: stream.alloc_zeros::<u32>(e)?,
            gathered: alloc_zeros_nonempty::<u32>(stream, n * e)?,
        };
        let outputs = raw::DispatchOutputs {
            routes: &mut out.routes,
            expert_counts: &mut out.expert_counts,
            gathered: &mut out.gathered,
        };
        self.launch(
            logits,
            shape,
            mode,
            layer_id,
            token_offset,
            log,
            Some(outputs),
        )?;
        Ok(out)
    }

    /// Id-range checks and the fused kernel launch (with or without the
    /// dispatch outputs).
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        logits: raw::LogitsArg<'_>,
        shape: BatchShape,
        mode: GateMode,
        layer_id: u16,
        token_offset: u32,
        log: &mut TelemetryBuffer,
        dispatch: Option<raw::DispatchOutputs<'_>>,
    ) -> Result<(), GpuError> {
        if shape.num_experts as usize > u16::MAX as usize + 1 {
            return Err(TelemetryError::ExpertIdOverflow {
                expert_id: shape.num_experts - 1,
            }
            .into());
        }
        check_token_range(token_offset, shape.num_tokens as usize)?;
        raw::launch_route_dispatch_log(
            &self.stream,
            &self.dispatch_log_fn,
            raw::DispatchLaunch {
                logits,
                num_tokens: shape.num_tokens,
                num_experts: shape.num_experts,
                k: shape.k,
                mode: mode_code(mode),
                layer_id: layer_id as u32,
                token_offset,
                dispatch,
                log: &mut log.records,
                log_counters: &mut log.counters,
            },
        )?;
        Ok(())
    }
}

/// Device-resident result of [`GpuRouter::route_dispatch_log`].
///
/// Buffers are allocated with at least one element (the driver rejects
/// zero-byte allocations), so logical sizes are tracked separately.
pub struct DeviceDispatch {
    stream: Arc<CudaStream>,
    num_experts: usize,
    num_tokens: usize,
    k: usize,
    routes: CudaSlice<Route>,
    expert_counts: CudaSlice<u32>,
    gathered: CudaSlice<u32>,
}

impl DeviceDispatch {
    /// Copies the expert load histogram (tokens per expert) to host.
    pub fn expert_load(&self) -> Result<Vec<u32>, GpuError> {
        Ok(self.stream.clone_dtoh(&self.expert_counts)?)
    }

    /// Copies routes (`[token * k + j]`) and the dispatch plan to host.
    pub fn to_host(&self) -> Result<(Vec<Route>, DispatchPlan), GpuError> {
        let (n, e) = (self.num_tokens, self.num_experts);
        let routes = copy_prefix(&self.stream, &self.routes, n * self.k)?;
        let counts = self.stream.clone_dtoh(&self.expert_counts)?;
        let gathered = copy_prefix(&self.stream, &self.gathered, n * e)?;
        Ok((routes, DispatchPlan::from_parts(e, n, counts, gathered)))
    }
}

/// Fixed-capacity telemetry log in GPU memory.
///
/// Kernels append to it with an atomic cursor; when it is full, further
/// records are counted as dropped instead of written. [`drain`](Self::drain)
/// is the only host transfer.
pub struct TelemetryBuffer {
    stream: Arc<CudaStream>,
    records: CudaSlice<TelemetryRecord>,
    /// `[append cursor, dropped]`. The cursor keeps counting past capacity,
    /// so `min(cursor, capacity)` is the number of valid records.
    counters: CudaSlice<u64>,
}

impl TelemetryBuffer {
    /// Allocates room for `capacity` records (12 bytes each) on `router`'s GPU.
    pub fn new(router: &GpuRouter, capacity: usize) -> Result<Self, GpuError> {
        if capacity == 0 {
            return Err(GpuError::Limit("telemetry capacity must be >= 1".into()));
        }
        let stream = router.stream.clone();
        Ok(TelemetryBuffer {
            records: stream.alloc_zeros::<TelemetryRecord>(capacity)?,
            counters: stream.alloc_zeros::<u64>(2)?,
            stream,
        })
    }

    /// Maximum records held between drains.
    pub fn capacity(&self) -> usize {
        self.records.len()
    }

    /// Waits for all logging kernels, copies the valid records and the
    /// dropped count to host, and resets the log to empty.
    ///
    /// Transfers: one 16-byte copy of the counters, then one copy of exactly
    /// the valid prefix of the record buffer (not the whole capacity).
    pub fn drain(&mut self) -> Result<Drained, GpuError> {
        self.stream.synchronize()?;
        let counters = self.stream.clone_dtoh(&self.counters)?;
        let (cursor, dropped) = (counters[0], counters[1]);
        let valid = cursor.min(self.capacity() as u64) as usize;
        let records = copy_prefix(&self.stream, &self.records, valid)?;
        self.stream.memset_zeros(&mut self.counters)?;
        Ok(Drained { records, dropped })
    }
}
