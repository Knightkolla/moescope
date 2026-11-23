//! GPU path: CUDA C kernels compiled at runtime with NVRTC, launched via cudarc.

mod raw;

pub use raw::cuda_available;

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, DriverError};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};

/// Anything that can go wrong on the GPU path.
#[derive(Debug)]
pub enum GpuError {
    /// `libcuda` or NVRTC could not be loaded (no NVIDIA driver / toolkit).
    CudaUnavailable,
    /// CUDA driver call failed (no GPU, out of memory, bad launch, ...).
    Driver(DriverError),
    /// NVRTC rejected the kernel source.
    Compile(CompileError),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::CudaUnavailable => write!(f, "CUDA driver or NVRTC library not found"),
            GpuError::Driver(e) => write!(f, "CUDA driver error: {e:?}"),
            GpuError::Compile(e) => write!(f, "NVRTC compile error: {e:?}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::Driver(e)
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
