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
