//! Phase 2 smoke test: proves the NVRTC -> load -> launch -> copy-back loop
//! works on this machine before any MoE kernels are involved.
//!
//! Run: cargo run --release --example vector_add --features cuda-13000

use cudarc::driver::{LaunchConfig, PushKernelArg};
use moescope::gpu::{GpuError, compile_module, open_context};

const SRC: &str = r#"
extern "C" __global__ void vector_add(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"#;

fn main() -> Result<(), GpuError> {
    let n: u32 = 1 << 20;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();

    let ctx = open_context(0)?;
    let (major, minor) = ctx.compute_capability()?;
    println!("device 0: {} (sm_{major}{minor})", ctx.name()?);

    let func = compile_module(&ctx, SRC)?.load_function("vector_add")?;
    let stream = ctx.default_stream();
    let a_dev = stream.clone_htod(&a)?;
    let b_dev = stream.clone_htod(&b)?;
    let mut out_dev = stream.alloc_zeros::<f32>(n as usize)?;

    let mut launch = stream.launch_builder(&func);
    launch.arg(&a_dev).arg(&b_dev).arg(&mut out_dev).arg(&n);
    // SAFETY: the kernel signature is (const float*, const float*, float*, unsigned int)
    // and the args match it in order and type. All three buffers hold exactly `n`
    // f32s and the kernel bounds-checks `i < n`, so no access is out of range.
    unsafe { launch.launch(LaunchConfig::for_num_elems(n)) }?;

    let out = stream.clone_dtoh(&out_dev)?;
    let bad = (0..n as usize)
        .filter(|&i| out[i] != 3.0 * i as f32)
        .count();
    if bad == 0 {
        println!("vector_add OK: {n} elements, out[12345] = {}", out[12345]);
        Ok(())
    } else {
        eprintln!("vector_add FAILED: {bad} wrong elements");
        std::process::exit(1)
    }
}
