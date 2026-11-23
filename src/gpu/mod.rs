//! GPU path: CUDA C kernels compiled at runtime with NVRTC, launched via cudarc.

pub mod kernels;
mod raw;

pub use raw::cuda_available;

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};

use crate::router::{GateMode, Route, RouteError, validate_batch};

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
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::CudaUnavailable => write!(f, "CUDA driver or NVRTC library not found"),
            GpuError::Driver(e) => write!(f, "CUDA driver error: {e:?}"),
            GpuError::Compile(e) => write!(f, "NVRTC compile error: {e:?}"),
            GpuError::Route(e) => write!(f, "{e}"),
            GpuError::Limit(msg) => write!(f, "GPU kernel limit: {msg}"),
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
    if k > kernels::MAX_K {
        return Err(GpuError::Limit(format!("k = {k} exceeds MAX_K = {}", kernels::MAX_K)));
    }
    to_u32(num_tokens * k, "num_tokens * k")?;
    to_u32(num_tokens * num_experts, "num_tokens * num_experts")?;
    Ok(BatchShape { num_tokens: to_u32(num_tokens, "num_tokens")?, num_experts: to_u32(num_experts, "num_experts")?, k: k as u32 })
}

/// GPU MoE router: owns the CUDA context, a stream, and the compiled kernels.
///
/// Create once per process (NVRTC compilation takes ~100 ms) and reuse.
pub struct GpuRouter {
    stream: Arc<CudaStream>,
    route_fn: CudaFunction,
}

impl GpuRouter {
    /// Opens GPU `ordinal` and compiles the routing kernels for it.
    pub fn new(ordinal: usize) -> Result<Self, GpuError> {
        let ctx = open_context(ordinal)?;
        let module = compile_module(&ctx, &kernels::source())?;
        Ok(GpuRouter { route_fn: module.load_function(kernels::ROUTE_KERNEL)?, stream: ctx.default_stream() })
    }

    /// Routes a `[num_tokens, num_experts]` batch on the GPU and copies the
    /// result back. Same contract and `[token * k + j]` layout as
    /// [`crate::router::route_batch`]; that function is its test oracle.
    pub fn route(&self, logits: &[f32], num_experts: usize, k: usize, mode: GateMode) -> Result<Vec<Route>, GpuError> {
        let shape = validate_for_gpu(logits, num_experts, k)?;
        if shape.num_tokens == 0 {
            return Ok(Vec::new());
        }
        let logits_dev = self.stream.clone_htod(logits)?;
        let mut out = self.stream.alloc_zeros::<Route>(shape.num_tokens as usize * k)?;
        raw::launch_route(&self.stream, &self.route_fn, &logits_dev, shape.num_tokens, shape.num_experts, shape.k, mode_code(mode), &mut out)?;
        Ok(self.stream.clone_dtoh(&out)?)
    }
}
