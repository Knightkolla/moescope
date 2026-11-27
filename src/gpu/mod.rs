//! GPU path: CUDA C kernels compiled at runtime with NVRTC, launched via cudarc.

pub mod kernels;
mod raw;

pub use raw::{ForeignLogits, ForeignPtrError, LogitsDtype, cuda_available};

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};

use crate::dispatcher::DispatchPlan;
use crate::router::{GateMode, Gating, Route, RouteError, check_shape, validate_batch};
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

/// A per-expert selection bias held on the GPU (sigmoid mode; DeepSeek-V3's
/// `e_score_correction_bias`). Upload once per layer with
/// [`GpuRouter::upload_bias`] and reuse it for every batch.
pub struct DeviceBias {
    buf: CudaSlice<f32>,
    /// The model keeps this bias in half precision (see
    /// [`GpuRouter::upload_bias_half`]).
    half: bool,
}

impl DeviceBias {
    /// Number of experts this bias covers.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Always false (a bias covers at least one expert).
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// Gating for one GPU call: a [`Gating`] configuration plus an optional
/// device-resident selection bias. A bare [`GateMode`] or [`Gating`]
/// converts into this, so simple callers never name it.
#[derive(Clone, Copy)]
pub struct GpuGate<'a> {
    pub gating: Gating,
    pub bias: Option<&'a DeviceBias>,
}

impl From<GateMode> for GpuGate<'_> {
    fn from(mode: GateMode) -> Self {
        Gating::from(mode).into()
    }
}

impl From<Gating> for GpuGate<'_> {
    fn from(gating: Gating) -> Self {
        GpuGate { gating, bias: None }
    }
}

impl<'a> GpuGate<'a> {
    /// The same configuration checks as the CPU router.
    fn validate(&self, num_experts: usize, k: usize) -> Result<(), GpuError> {
        self.gating.validate(num_experts, k, None)?;
        if let Some(b) = self.bias {
            if self.gating.mode != GateMode::Sigmoid {
                return Err(RouteError::InvalidGating(
                    "a selection bias needs mode 'sigmoid'".into(),
                )
                .into());
            }
            if b.len() != num_experts {
                return Err(RouteError::InvalidGating(format!(
                    "bias has {} entries, layer has {num_experts} experts",
                    b.len()
                ))
                .into());
            }
        }
        Ok(())
    }

    fn arg(&self) -> raw::GateArg<'a> {
        raw::GateArg {
            gating: self.gating,
            bias: self.bias.map(|b| &b.buf),
            round_biased: self.bias.is_some_and(|b| b.half),
        }
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

/// A host dispatch mask must have one byte per (token, expert).
fn check_kept_len(len: Option<usize>, shape: BatchShape) -> Result<(), GpuError> {
    let expected = shape.num_tokens as usize * shape.num_experts as usize;
    match len {
        Some(found) if found != expected => {
            Err(TelemetryError::KeptMaskShape { expected, found }.into())
        }
        _ => Ok(()),
    }
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

/// The kernels of one compiled variant (logits dtype x softmax/sigmoid).
struct Kernels {
    route_fn: CudaFunction,
    dispatch_log_fn: CudaFunction,
    advance_fn: CudaFunction,
}

/// GPU MoE router: owns the CUDA context, a stream, and the compiled kernels.
///
/// Create once per process and reuse. The kernels are compiled with NVRTC
/// once per variant (~100 ms each): float32 softmax up front, the others
/// (float16 / bfloat16 logits, sigmoid gating) on first use.
pub struct GpuRouter {
    stream: Arc<CudaStream>,
    variants: [OnceLock<Kernels>; kernels::NUM_VARIANTS],
}

impl GpuRouter {
    /// Opens GPU `ordinal` and compiles the routing kernels for it.
    pub fn new(ordinal: usize) -> Result<Self, GpuError> {
        let ctx = open_context(ordinal)?;
        let router = GpuRouter {
            stream: ctx.default_stream(),
            variants: Default::default(),
        };
        router.kernels(LogitsDtype::F32, false)?;
        Ok(router)
    }

    /// The compiled kernels for logits of `dtype` and (non-)sigmoid gating.
    fn kernels(&self, dtype: LogitsDtype, sigmoid: bool) -> Result<&Kernels, GpuError> {
        let code = match dtype {
            LogitsDtype::F32 => 0,
            LogitsDtype::F16 => 1,
            LogitsDtype::BF16 => 2,
        };
        let slot = &self.variants[kernels::variant(code, sigmoid)];
        if let Some(k) = slot.get() {
            return Ok(k);
        }
        let module = compile_module(
            self.stream.context(),
            &kernels::source(kernels::variant(code, sigmoid)),
        )?;
        let compiled = Kernels {
            route_fn: module.load_function(kernels::ROUTE_KERNEL)?,
            dispatch_log_fn: module.load_function(kernels::DISPATCH_LOG_KERNEL)?,
            advance_fn: module.load_function(kernels::ADVANCE_TOKENS_KERNEL)?,
        };
        Ok(slot.get_or_init(|| compiled))
    }

    /// Copies a per-expert selection bias to the GPU once, for reuse with
    /// every batch of that layer (see [`GpuGate`]). Values must be finite.
    pub fn upload_bias(&self, bias: &[f32]) -> Result<DeviceBias, GpuError> {
        if bias.is_empty() {
            return Err(RouteError::InvalidGating("bias is empty".into()).into());
        }
        if let Some(i) = bias.iter().position(|x| !x.is_finite()) {
            return Err(RouteError::InvalidGating(format!(
                "bias[{i}] = {} is not finite",
                bias[i]
            ))
            .into());
        }
        Ok(DeviceBias {
            buf: self.stream.clone_htod(bias)?,
            half: false,
        })
    }

    /// [`upload_bias`](Self::upload_bias) for a model that keeps the bias
    /// in the same half precision as its logits: score + bias is then
    /// rounded to that precision before ranking, exactly as torch adds two
    /// half tensors. (No effect on float32 logits.)
    pub fn upload_bias_half(&self, bias: &[f32]) -> Result<DeviceBias, GpuError> {
        Ok(DeviceBias {
            half: true,
            ..self.upload_bias(bias)?
        })
    }

    /// Routes a `[num_tokens, num_experts]` batch on the GPU and copies the
    /// result back. Same contract and `[token * k + j]` layout as
    /// [`crate::router::route_batch`]; that function is its test oracle.
    pub fn route<'g>(
        &self,
        logits: &[f32],
        num_experts: usize,
        k: usize,
        gate: impl Into<GpuGate<'g>>,
    ) -> Result<Vec<Route>, GpuError> {
        let gate = gate.into();
        let shape = validate_for_gpu(logits, num_experts, k)?;
        gate.validate(num_experts, k)?;
        if shape.num_tokens == 0 {
            return Ok(Vec::new());
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        let mut out = self
            .stream
            .alloc_zeros::<Route>(shape.num_tokens as usize * k)?;
        raw::launch_route(
            &self.stream,
            &self
                .kernels(LogitsDtype::F32, gate.gating.mode == GateMode::Sigmoid)?
                .route_fn,
            &logits_dev,
            shape.num_tokens,
            shape.num_experts,
            shape.k,
            gate.arg(),
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
    pub fn route_dispatch_log<'g>(
        &self,
        logits: &[f32],
        num_experts: usize,
        k: usize,
        gate: impl Into<GpuGate<'g>>,
        layer_id: u8,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<DeviceDispatch, GpuError> {
        let gate = gate.into();
        let shape = validate_for_gpu(logits, num_experts, k)?;
        gate.validate(num_experts, k)?;
        if shape.num_tokens == 0 {
            return self.empty_dispatch(num_experts);
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        self.launch_dispatch(
            raw::LogitsArg::Owned(&logits_dev),
            shape,
            gate,
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
    pub fn route_dispatch_log_device<'g>(
        &self,
        logits_ptr: u64,
        num_tokens: usize,
        num_experts: usize,
        k: usize,
        gate: impl Into<GpuGate<'g>>,
        layer_id: u8,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<DeviceDispatch, GpuError> {
        let gate = gate.into();
        let shape = validate_shape_for_gpu(num_tokens, num_experts, k)?;
        gate.validate(num_experts, k)?;
        if num_tokens == 0 {
            return self.empty_dispatch(num_experts);
        }
        let foreign = raw::borrow_foreign(
            self.stream.context(),
            logits_ptr,
            num_tokens * num_experts,
            LogitsDtype::F32,
        )?
        .map_err(GpuError::ForeignPtr)?;
        self.launch_dispatch(
            raw::LogitsArg::Foreign(foreign),
            shape,
            gate,
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
    pub fn log_only<'g>(
        &self,
        logits: &[f32],
        num_experts: usize,
        k: usize,
        gate: impl Into<GpuGate<'g>>,
        layer_id: u8,
        token_offset: u32,
        log: &mut TelemetryBuffer,
    ) -> Result<(), GpuError> {
        let gate = gate.into();
        let shape = validate_for_gpu(logits, num_experts, k)?;
        gate.validate(num_experts, k)?;
        if shape.num_tokens == 0 {
            return Ok(());
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        let logits = raw::LogitsArg::Owned(&logits_dev);
        self.launch(logits, shape, gate, layer_id, token_offset, log, None, None)
    }

    /// [`log_only`](Self::log_only) plus the model's dispatch mask after
    /// capacity (`kept[t * num_experts + e] != 0` iff expert `e` computed
    /// token `t`): decisions the experts never computed are logged with
    /// [`CAPACITY_DROP`](crate::telemetry::CAPACITY_DROP) set.
    #[allow(clippy::too_many_arguments)]
    pub fn log_only_masked<'g>(
        &self,
        logits: &[f32],
        num_experts: usize,
        k: usize,
        gate: impl Into<GpuGate<'g>>,
        layer_id: u8,
        token_offset: u32,
        kept: Option<&[u8]>,
        log: &mut TelemetryBuffer,
    ) -> Result<(), GpuError> {
        let gate = gate.into();
        let shape = validate_for_gpu(logits, num_experts, k)?;
        gate.validate(num_experts, k)?;
        check_kept_len(kept.map(<[u8]>::len), shape)?;
        if shape.num_tokens == 0 {
            return Ok(());
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        let kept_dev = kept.map(|m| self.stream.clone_htod(m)).transpose()?;
        let logits = raw::LogitsArg::Owned(&logits_dev);
        let kept = kept_dev.as_ref().map(raw::MaskArg::Owned);
        self.launch(logits, shape, gate, layer_id, token_offset, log, None, kept)
    }

    /// Zero-copy [`log_only`](Self::log_only) for logits already on the GPU,
    /// in float32, float16 or bfloat16 (same pointer validation and caller
    /// requirements as [`route_dispatch_log_device`](Self::route_dispatch_log_device)).
    /// This is what the Python `route_and_log` uses for CUDA tensors: one
    /// pointer check plus one kernel launch per call.
    ///
    /// `stream` is a raw `CUstream` handle of this GPU's context to launch on
    /// (torch: `torch.cuda.current_stream().cuda_stream`), or 0 for the
    /// legacy default stream. Launching on the model's own stream needs no
    /// host synchronization and works inside CUDA-graph capture.
    ///
    /// `kept_ptr`, if not 0, is the device address of the model's u8
    /// dispatch mask after capacity, `[num_tokens, num_experts]` (see
    /// [`log_only_masked`](Self::log_only_masked)), validated like the logits.
    #[allow(clippy::too_many_arguments)]
    pub fn log_only_device<'g>(
        &self,
        logits_ptr: u64,
        dtype: LogitsDtype,
        num_tokens: usize,
        num_experts: usize,
        k: usize,
        gate: impl Into<GpuGate<'g>>,
        layer_id: u8,
        token_offset: u32,
        log: &mut TelemetryBuffer,
        stream: u64,
        kept_ptr: u64,
    ) -> Result<(), GpuError> {
        let gate = gate.into();
        let shape = validate_shape_for_gpu(num_tokens, num_experts, k)?;
        gate.validate(num_experts, k)?;
        if num_tokens == 0 {
            return Ok(());
        }
        let ctx = self.stream.context();
        let foreign = raw::borrow_foreign(ctx, logits_ptr, num_tokens * num_experts, dtype)?
            .map_err(GpuError::ForeignPtr)?;
        let kept = match kept_ptr {
            0 => None,
            p => Some(
                raw::borrow_foreign_bytes(ctx, p, num_tokens * num_experts)?
                    .map_err(GpuError::ForeignPtr)?,
            ),
        };
        let borrowed = raw::borrow_stream(ctx, stream)?.map_err(GpuError::ForeignPtr)?;
        let stream = borrowed.as_ref().unwrap_or(&self.stream);
        let logits = raw::LogitsArg::Foreign(foreign);
        self.launch_on(
            stream,
            logits,
            shape,
            gate,
            layer_id,
            token_offset,
            log,
            None,
            kept,
        )
    }

    /// Adds `n` to `log`'s device-side token counter, on `stream` (a raw
    /// `CUstream` handle, 0 = legacy default). Call it once after each
    /// forward pass when the log uses device token ids
    /// ([`TelemetryBuffer::set_device_token_ids`]); inside CUDA-graph
    /// capture it is recorded and replays with the graph.
    pub fn advance_tokens(
        &self,
        log: &mut TelemetryBuffer,
        n: u64,
        stream: u64,
    ) -> Result<(), GpuError> {
        let borrowed =
            raw::borrow_stream(self.stream.context(), stream)?.map_err(GpuError::ForeignPtr)?;
        let stream = borrowed.as_ref().unwrap_or(&self.stream);
        let f = &self.kernels(LogitsDtype::F32, false)?.advance_fn;
        raw::launch_advance_tokens(stream, f, &mut log.tokens, n)?;
        Ok(())
    }

    /// Shared tail of the dispatch entry points: allocate the per-call
    /// outputs, then launch.
    fn launch_dispatch(
        &self,
        logits: raw::LogitsArg<'_>,
        shape: BatchShape,
        gate: GpuGate<'_>,
        layer_id: u8,
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
            gate,
            layer_id,
            token_offset,
            log,
            Some(outputs),
            None,
        )?;
        Ok(out)
    }

    /// [`launch_on`](Self::launch_on) the router's own default stream.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        logits: raw::LogitsArg<'_>,
        shape: BatchShape,
        gate: GpuGate<'_>,
        layer_id: u8,
        token_offset: u32,
        log: &mut TelemetryBuffer,
        dispatch: Option<raw::DispatchOutputs<'_>>,
        kept: Option<raw::MaskArg<'_>>,
    ) -> Result<(), GpuError> {
        let stream = self.stream.clone();
        self.launch_on(
            &stream,
            logits,
            shape,
            gate,
            layer_id,
            token_offset,
            log,
            dispatch,
            kept,
        )
    }

    /// Id-range checks and the fused kernel launch (with or without the
    /// dispatch outputs).
    #[allow(clippy::too_many_arguments)]
    fn launch_on(
        &self,
        stream: &Arc<CudaStream>,
        logits: raw::LogitsArg<'_>,
        shape: BatchShape,
        gate: GpuGate<'_>,
        layer_id: u8,
        token_offset: u32,
        log: &mut TelemetryBuffer,
        dispatch: Option<raw::DispatchOutputs<'_>>,
        kept: Option<raw::MaskArg<'_>>,
    ) -> Result<(), GpuError> {
        if shape.num_experts as usize > u16::MAX as usize + 1 {
            return Err(TelemetryError::ExpertIdOverflow {
                expert_id: shape.num_experts - 1,
            }
            .into());
        }
        check_token_range(token_offset, shape.num_tokens as usize)?;
        let kernels = self.kernels(logits.dtype(), gate.gating.mode == GateMode::Sigmoid)?;
        let TelemetryBuffer {
            records,
            counters,
            tokens,
            device_tokens,
            ..
        } = log;
        raw::launch_route_dispatch_log(
            stream,
            &kernels.dispatch_log_fn,
            raw::DispatchLaunch {
                logits,
                num_tokens: shape.num_tokens,
                num_experts: shape.num_experts,
                k: shape.k,
                gate: gate.arg(),
                token_base: device_tokens.then_some(&*tokens),
                layer_id: layer_id as u32,
                token_offset,
                dispatch,
                kept,
                log: records,
                log_counters: counters,
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
    /// Device-side token counter, added to token ids when `device_tokens`.
    tokens: CudaSlice<u64>,
    device_tokens: bool,
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
            tokens: stream.alloc_zeros::<u64>(1)?,
            device_tokens: false,
            stream,
        })
    }

    /// Maximum records held between drains.
    pub fn capacity(&self) -> usize {
        self.records.len()
    }

    /// Turns device-side token ids on or off. When on, every record's
    /// `token_id` is `device counter + token_offset + token index`, and the
    /// counter only moves via [`GpuRouter::advance_tokens`]. That keeps ids
    /// correct when logging is replayed from a captured CUDA graph, where
    /// host-side `token_offset` values are frozen at capture time.
    pub fn set_device_token_ids(&mut self, on: bool) {
        self.device_tokens = on;
    }

    /// Current value of the device token counter (synchronizes).
    pub fn device_token_count(&self) -> Result<u64, GpuError> {
        self.stream.context().synchronize()?;
        Ok(self.stream.clone_dtoh(&self.tokens)?[0])
    }

    /// Waits for all logging kernels, copies the valid records and the
    /// dropped count to host, and resets the log to empty.
    ///
    /// Transfers: one 16-byte copy of the counters, then one copy of exactly
    /// the valid prefix of the record buffer (not the whole capacity).
    pub fn drain(&mut self) -> Result<Drained, GpuError> {
        // Whole-context sync: logging may have run on other (borrowed) streams.
        self.stream.context().synchronize()?;
        let counters = self.stream.clone_dtoh(&self.counters)?;
        let (cursor, dropped) = (counters[0], counters[1]);
        let valid = cursor.min(self.capacity() as u64) as usize;
        let records = copy_prefix(&self.stream, &self.records, valid)?;
        self.stream.memset_zeros(&mut self.counters)?;
        Ok(Drained { records, dropped })
    }
}
