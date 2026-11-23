//! Python bindings (feature `python`), built by maturin as `moescope._native`.
//!
//! The user-facing API (`route_and_log`, `drain`, numpy dtype, torch
//! handling) lives in `python/moescope/__init__.py`; this module exposes one
//! class, `Recorder`, that owns a router plus a telemetry log on either the
//! CPU or the GPU.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::router::{GateMode, route_batch};
use crate::telemetry::{Drained, HostTelemetry, TelemetryRecord};

#[cfg(feature = "cuda")]
use crate::gpu::{GpuError, GpuRouter, TelemetryBuffer};

/// Where routing runs and where the log lives.
enum Backend {
    /// Reference router + bounded host log. Always available.
    Cpu(HostTelemetry),
    /// Fused CUDA kernel + device-resident log.
    #[cfg(feature = "cuda")]
    Gpu {
        router: GpuRouter,
        log: TelemetryBuffer,
    },
}

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[cfg(feature = "cuda")]
fn gpu_err(e: GpuError) -> PyErr {
    match e {
        GpuError::Route(_)
        | GpuError::Telemetry(_)
        | GpuError::Limit(_)
        | GpuError::ForeignPtr(_) => value_err(e),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn parse_mode(mode: &str) -> PyResult<GateMode> {
    match mode {
        "softmax_topk" => Ok(GateMode::SoftmaxThenTopK),
        "topk_softmax" => Ok(GateMode::TopKThenSoftmax),
        other => Err(value_err(format!(
            "mode must be 'softmax_topk' or 'topk_softmax', got {other:?}"
        ))),
    }
}

/// `"cpu"`, `"cuda"`, `"cuda:N"` or `"auto"` (GPU 0 if available, else CPU).
fn open_backend(device: &str, capacity: usize) -> PyResult<Backend> {
    let cpu = || Ok(Backend::Cpu(HostTelemetry::new(capacity)));
    let ordinal = match device {
        "cpu" => return cpu(),
        "auto" if !native_cuda_available() => return cpu(),
        "auto" | "cuda" => 0,
        d => match d
            .strip_prefix("cuda:")
            .and_then(|n| n.parse::<usize>().ok())
        {
            Some(n) => n,
            None => return Err(value_err(format!("unknown device {d:?}"))),
        },
    };
    open_gpu(ordinal, capacity)
}

#[cfg(feature = "cuda")]
fn open_gpu(ordinal: usize, capacity: usize) -> PyResult<Backend> {
    let router = GpuRouter::new(ordinal).map_err(gpu_err)?;
    let log = TelemetryBuffer::new(&router, capacity).map_err(gpu_err)?;
    Ok(Backend::Gpu { router, log })
}

#[cfg(not(feature = "cuda"))]
fn open_gpu(_ordinal: usize, _capacity: usize) -> PyResult<Backend> {
    Err(PyRuntimeError::new_err(
        "moescope was built without the 'cuda' feature",
    ))
}

fn native_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    return crate::gpu::cuda_available();
    #[cfg(not(feature = "cuda"))]
    false
}

/// Serializes records as little-endian bytes matching the numpy dtype
/// `[('token_id','<u4'),('layer_id','<u2'),('expert_id','<u2'),('gate','<f4')]`.
fn records_to_bytes(records: &[TelemetryRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(records));
    for r in records {
        out.extend_from_slice(&r.token_id.to_le_bytes());
        out.extend_from_slice(&r.layer_id.to_le_bytes());
        out.extend_from_slice(&r.expert_id.to_le_bytes());
        out.extend_from_slice(&r.gate.to_le_bytes());
    }
    out
}

/// Router + telemetry log. One per process is typical; the Python package
/// keeps a default instance behind `moescope.route_and_log` / `drain`.
#[pyclass(module = "moescope._native", unsendable)]
pub struct Recorder {
    backend: Backend,
    mode: GateMode,
    capacity: usize,
}

#[pymethods]
impl Recorder {
    #[new]
    #[pyo3(signature = (capacity = 1 << 22, mode = "softmax_topk", device = "auto"))]
    fn new(capacity: usize, mode: &str, device: &str) -> PyResult<Self> {
        if capacity == 0 {
            return Err(value_err("capacity must be >= 1"));
        }
        Ok(Recorder {
            backend: open_backend(device, capacity)?,
            mode: parse_mode(mode)?,
            capacity,
        })
    }

    /// "cpu" or "cuda".
    #[getter]
    fn backend(&self) -> &'static str {
        match self.backend {
            Backend::Cpu(_) => "cpu",
            #[cfg(feature = "cuda")]
            Backend::Gpu { .. } => "cuda",
        }
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.capacity
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.mode {
            GateMode::SoftmaxThenTopK => "softmax_topk",
            GateMode::TopKThenSoftmax => "topk_softmax",
        }
    }

    /// Routes host logits (any C-contiguous float32 buffer of shape
    /// `[num_tokens, num_experts]`) and appends every decision to the log.
    #[pyo3(signature = (logits, k, layer_id, token_offset = 0))]
    fn route_and_log(
        &mut self,
        py: Python<'_>,
        logits: PyBuffer<f32>,
        k: usize,
        layer_id: u16,
        token_offset: u32,
    ) -> PyResult<()> {
        let &[num_tokens, num_experts] = logits.shape() else {
            return Err(value_err(format!(
                "logits must be 2-D [tokens, experts], got shape {:?}",
                logits.shape()
            )));
        };
        if !logits.is_c_contiguous() {
            return Err(value_err("logits must be C-contiguous"));
        }
        let host = logits.to_vec(py)?;
        debug_assert_eq!(host.len(), num_tokens * num_experts);
        let mode = self.mode;
        let backend = &mut self.backend;
        py.detach(move || match backend {
            Backend::Cpu(log) => {
                let routes = route_batch(&host, num_experts, k, mode).map_err(value_err)?;
                log.log_routes(&routes, k, layer_id, token_offset)
                    .map_err(value_err)
            }
            #[cfg(feature = "cuda")]
            Backend::Gpu { router, log } => router
                .route_dispatch_log(&host, num_experts, k, mode, layer_id, token_offset, log)
                .map(drop)
                .map_err(gpu_err),
        })
    }

    /// Zero-copy variant for logits already on the GPU: `ptr` is the device
    /// address of a C-contiguous float32 `[num_tokens, num_experts]` tensor
    /// (torch: `t.data_ptr()`). The pointer is validated with the CUDA
    /// driver before use. GPU backend only.
    #[pyo3(signature = (ptr, num_tokens, num_experts, k, layer_id, token_offset = 0))]
    #[allow(clippy::too_many_arguments)]
    fn route_and_log_device(
        &mut self,
        py: Python<'_>,
        ptr: u64,
        num_tokens: usize,
        num_experts: usize,
        k: usize,
        layer_id: u16,
        token_offset: u32,
    ) -> PyResult<()> {
        let mode = self.mode;
        match &mut self.backend {
            Backend::Cpu(_) => {
                let _ = (
                    py,
                    ptr,
                    num_tokens,
                    num_experts,
                    k,
                    layer_id,
                    token_offset,
                    mode,
                );
                Err(PyRuntimeError::new_err(
                    "route_and_log_device needs the CUDA backend",
                ))
            }
            #[cfg(feature = "cuda")]
            Backend::Gpu { router, log } => py.detach(move || {
                router
                    .route_dispatch_log_device(
                        ptr,
                        num_tokens,
                        num_experts,
                        k,
                        mode,
                        layer_id,
                        token_offset,
                        log,
                    )
                    .map(drop)
                    .map_err(gpu_err)
            }),
        }
    }

    /// Copies the whole log to host once and resets it. Returns
    /// `(records_bytes, dropped)`; the Python wrapper views the bytes as a
    /// numpy structured array.
    fn drain<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, u64)> {
        let backend = &mut self.backend;
        let Drained { records, dropped } = py.detach(move || -> PyResult<Drained> {
            match backend {
                Backend::Cpu(log) => Ok(log.drain()),
                #[cfg(feature = "cuda")]
                Backend::Gpu { log, .. } => log.drain().map_err(gpu_err),
            }
        })?;
        Ok((PyBytes::new(py, &records_to_bytes(&records)), dropped))
    }
}

/// True if this build has the CUDA backend and a driver is present.
#[pyfunction]
fn cuda_available() -> bool {
    native_cuda_available()
}

/// True if this build was compiled with the `cuda` feature.
#[pyfunction]
fn built_with_cuda() -> bool {
    cfg!(feature = "cuda")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Recorder>()?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(built_with_cuda, m)?)?;
    m.add("RECORD_ITEMSIZE", std::mem::size_of::<TelemetryRecord>())?;
    Ok(())
}
