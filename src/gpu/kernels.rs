//! CUDA C source for the MoE kernels, compiled at runtime by NVRTC.
//!
//! The source deliberately avoids every header (NVRTC has no libc), so it
//! uses `unsigned int` etc. directly and builds -inf from its bit pattern.
//! The struct layouts here must match the `#[repr(C)]` Rust types:
//! `RouteEntry` = [`crate::router::Route`],
//! `TelemetryRecord` = [`crate::telemetry::TelemetryRecord`].

/// Largest `k` the kernels support. Each lane keeps the token's top-k in a
/// fixed-size local array, so this is its bound.
pub const MAX_K: usize = 32;

/// Kernel entry-point names, kept next to the source that defines them.
pub const ROUTE_KERNEL: &str = "moe_route";
pub const DISPATCH_LOG_KERNEL: &str = "moe_route_dispatch_log";

/// Lanes that cooperate on one token: the smallest power of two covering
/// `num_experts / 4`, clamped to 1..=32. Each lane then scans about four
/// experts per pass, reads of one row are contiguous across the group, and
/// a group never straddles a warp (32 is a multiple of every size).
/// The host sizes the grid with this and passes it to the kernels.
pub fn lanes_per_token(num_experts: u32) -> u32 {
    num_experts.div_ceil(4).next_power_of_two().clamp(1, 32)
}

/// Kernel source. `MOESCOPE_MAX_K` is substituted from [`MAX_K`] by [`source`].
const SRC: &str = r#"
#define MAX_K MOESCOPE_MAX_K
#define FULL_MASK 0xffffffffu
#define NO_EXPERT 0xffffffffu

// Mirrors crate::router::Route.
struct RouteEntry {
    unsigned int expert_id;
    float gate;
};

// Mirrors crate::telemetry::TelemetryRecord (12 bytes, no padding).
struct TelemetryRecord {
    unsigned int token_id;
    unsigned short layer_id;
    unsigned short expert_id;
    float gate;
};

// NaN is ranked as -inf, exactly like the CPU router's rank_order, so a NaN
// logit can never win a slot over a real number.
__device__ __forceinline__ float rank_key(float x) {
    return (x != x) ? __int_as_float(0xff800000) : x;
}

// The routing order, same as the CPU sort: higher logit first, and on an
// exact tie the lower expert index first. (-inf, NO_EXPERT) ranks after
// every real expert, so it doubles as "no candidate".
__device__ __forceinline__ bool ranks_before(float a, unsigned int ia, float b, unsigned int ib) {
    return a > b || (a == b && ia < ib);
}

// One lane's best expert among its stripe (lane, lane+G, lane+2G, ...),
// restricted (when has_prev) to experts ranked strictly after (pv, pi).
__device__ void lane_best(const float* row, unsigned int num_experts, unsigned int lane,
                          unsigned int lanes, bool has_prev, float pv, unsigned int pi,
                          float* bv, unsigned int* bi) {
    float v = __int_as_float(0xff800000);
    unsigned int i = NO_EXPERT;
    for (unsigned int e = lane; e < num_experts; e += lanes) {
        float x = rank_key(row[e]);
        if (has_prev && !ranks_before(pv, pi, x, e)) continue;  // already taken
        if (ranks_before(x, e, v, i)) { v = x; i = e; }
    }
    *bv = v;
    *bi = i;
}

// Top-k for one token, computed by a group of `lanes` threads.
//
// Every lane keeps the best expert of its own stripe. Each of the k rounds
// is a butterfly (xor-shuffle) reduction, so all lanes learn the round's
// winner; only the lane that owned it rescans its stripe for its next best.
// Reads of a row are contiguous across the group, and the work per lane is
// about (E / lanes) * (1 + rounds it wins) instead of E * k.
//
// All 32 lanes of the warp must call this the same number of times (the
// shuffles use the full mask), so out-of-range tokens pass valid = false
// and take part with no candidates.
__device__ void group_top_k(const float* row, bool valid, unsigned int num_experts,
                            unsigned int k, unsigned int lane, unsigned int lanes,
                            unsigned int* ids, float* vals) {
    float bv;
    unsigned int bi;
    if (valid) lane_best(row, num_experts, lane, lanes, false, 0.0f, 0, &bv, &bi);
    else { bv = __int_as_float(0xff800000); bi = NO_EXPERT; }
    for (unsigned int j = 0; j < k; ++j) {
        float v = bv;
        unsigned int i = bi;
        for (unsigned int off = lanes >> 1; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(FULL_MASK, v, off, lanes);
            unsigned int oi = __shfl_xor_sync(FULL_MASK, i, off, lanes);
            if (ranks_before(ov, oi, v, i)) { v = ov; i = oi; }
        }
        vals[j] = v;
        ids[j] = i;
        if (valid && bi == i) lane_best(row, num_experts, lane, lanes, true, v, i, &bv, &bi);
    }
}

// Softmax denominator for the token (mode values match GateMode):
//   0 = SoftmaxThenTopK: sum over all E logits, shared across the group.
//   1 = TopKThenSoftmax: sum over the k selected logits only (Mixtral).
// vals[0] is the row max in both cases, which keeps exp() from overflowing.
__device__ float gate_denominator(const float* row, bool valid, unsigned int num_experts,
                                  unsigned int k, int mode, unsigned int lane,
                                  unsigned int lanes, const float* vals) {
    float max = vals[0];
    float sum = 0.0f;
    if (mode == 0) {
        if (valid)
            for (unsigned int e = lane; e < num_experts; e += lanes) sum += expf(rank_key(row[e]) - max);
        for (unsigned int off = lanes >> 1; off > 0; off >>= 1)
            sum += __shfl_xor_sync(FULL_MASK, sum, off, lanes);
    } else {
        for (unsigned int j = 0; j < k; ++j) sum += expf(vals[j] - max);
    }
    return sum;
}

// Which token this thread works on, and its lane within the token's group.
__device__ __forceinline__ void token_and_lane(unsigned int lanes, unsigned long long* t,
                                               unsigned int* lane) {
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    *t = gid / lanes;
    *lane = (unsigned int)(gid % lanes);
}

// Phase 3: route only. Writes out[t * k + j] (best expert first); lane l of
// the group writes slots j = l, l + lanes, ...
extern "C" __global__ void moe_route(const float* logits, unsigned int num_tokens,
                                     unsigned int num_experts, unsigned int k, int mode,
                                     RouteEntry* out, unsigned int lanes) {
    unsigned long long t;
    unsigned int lane;
    token_and_lane(lanes, &t, &lane);
    bool valid = t < num_tokens;
    const float* row = logits + (valid ? t : 0) * num_experts;
    unsigned int ids[MAX_K];
    float vals[MAX_K];
    group_top_k(row, valid, num_experts, k, lane, lanes, ids, vals);
    float sum = gate_denominator(row, valid, num_experts, k, mode, lane, lanes, vals);
    if (!valid) return;
    for (unsigned int j = lane; j < k; j += lanes) {
        out[t * k + j].expert_id = ids[j];
        out[t * k + j].gate = expf(vals[j] - vals[0]) / sum;
    }
}

// Phase 4: route + dispatch + telemetry in one pass, one lane group per token.
//
// Dispatch: atomicAdd on expert_counts[e] hands the token a unique slot in
// expert e's block of `gathered` (stride num_tokens, which can never
// overflow since a token picks each expert at most once). The final
// expert_counts is the expert load histogram.
//
// Telemetry: the group's lane 0 reserves the token's k consecutive records
// with ONE atomicAdd on log_counters[0] (the append cursor, which keeps
// counting past capacity) and shares the base with the group. Records that
// land at or beyond `capacity` are not written; they are counted in
// log_counters[1] (dropped). Nothing is copied to the host here.
//
// write_dispatch == 0 is the telemetry-only mode: routes, expert_counts and
// gathered are never touched (they may be null), so logging costs one launch
// and no allocations.
extern "C" __global__ void moe_route_dispatch_log(
        const float* logits, unsigned int num_tokens, unsigned int num_experts,
        unsigned int k, int mode, unsigned int layer_id, unsigned int token_offset,
        RouteEntry* routes, unsigned int* expert_counts, unsigned int* gathered,
        TelemetryRecord* log, unsigned long long capacity,
        unsigned long long* log_counters, int write_dispatch, unsigned int lanes) {
    unsigned long long t;
    unsigned int lane;
    token_and_lane(lanes, &t, &lane);
    bool valid = t < num_tokens;
    const float* row = logits + (valid ? t : 0) * num_experts;
    unsigned int ids[MAX_K];
    float vals[MAX_K];
    group_top_k(row, valid, num_experts, k, lane, lanes, ids, vals);
    float sum = gate_denominator(row, valid, num_experts, k, mode, lane, lanes, vals);

    unsigned long long base = 0;
    if (valid && lane == 0) base = atomicAdd(&log_counters[0], (unsigned long long)k);
    base = __shfl_sync(FULL_MASK, base, 0, lanes);
    if (!valid) return;

    unsigned int dropped = 0;
    for (unsigned int j = lane; j < k; j += lanes) {
        unsigned int e = ids[j];
        float gate = expf(vals[j] - vals[0]) / sum;
        if (write_dispatch) {
            unsigned int a = (unsigned int)t * k + j;
            routes[a].expert_id = e;
            routes[a].gate = gate;
            unsigned int slot = atomicAdd(&expert_counts[e], 1u);
            gathered[e * num_tokens + slot] = a;
        }

        unsigned long long idx = base + j;
        if (idx < capacity) {
            log[idx].token_id = token_offset + (unsigned int)t;
            log[idx].layer_id = (unsigned short)layer_id;
            log[idx].expert_id = (unsigned short)e;
            log[idx].gate = gate;
        } else {
            ++dropped;
        }
    }
    if (dropped > 0) atomicAdd(&log_counters[1], (unsigned long long)dropped);
}
"#;

/// Full kernel source with constants substituted, ready for NVRTC.
pub fn source() -> String {
    SRC.replace("MOESCOPE_MAX_K", &MAX_K.to_string())
}

#[cfg(test)]
mod tests {
    use super::lanes_per_token;

    #[test]
    fn lanes_are_a_power_of_two_dividing_a_warp() {
        for e in 1..=70_000u32 {
            let l = lanes_per_token(e);
            assert!(l.is_power_of_two() && l <= 32, "E={e} -> {l}");
        }
        let picks: Vec<_> = [1, 8, 32, 40, 64, 128, 256].iter().map(|&e| lanes_per_token(e)).collect();
        assert_eq!(picks, [1, 2, 8, 16, 16, 32, 32]);
    }
}
