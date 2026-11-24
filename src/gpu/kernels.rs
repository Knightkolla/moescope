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
pub const ADVANCE_TOKENS_KERNEL: &str = "moe_advance_tokens";

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
#define MAX_GROUPS MOESCOPE_MAX_GROUPS
// Compile-time variant (one NVRTC module per combination, see source()):
// DT = logits dtype (0 float32, 1 float16, 2 bfloat16); SIG = sigmoid mode.
// Keeping both out of the inner loops is what keeps the kernel fast.
#define DT MOESCOPE_DT
#define SIG MOESCOPE_SIG
#define FULL_MASK 0xffffffffu
#define NO_EXPERT 0xffffffffu
#define ALL_GROUPS 0xffffffffu

// Mirrors gpu::raw::GateParams (48 bytes). mode: 0 = softmax_topk,
// 1 = topk_softmax, 2 = sigmoid. num_groups == 1 means no group limit.
// logits_dtype: 0 = float32, 1 = float16, 2 = bfloat16.
// bias is a [num_experts] device array or null (sigmoid mode only).
// token_base, if not null, is a device counter added to every token id, so
// a captured CUDA graph logs fresh ids on each replay. round_biased: the
// model stores its bias in the logits' half precision, so score + bias is
// itself rounded to that precision before ranking (as torch does).
struct GateParams {
    int mode;
    unsigned int num_groups;
    unsigned int top_groups;
    int normalize;
    float scale;
    int logits_dtype;
    const float* bias;
    const unsigned long long* token_base;
    int round_biased;
    int pad;
};

// One token's logits, in the variant's precision DT.
struct Row {
    const char* p;
};

// Logit of expert e as float32. float16 is widened with one PTX convert
// (NVRTC has no cuda_fp16.h); bfloat16 is the top half of a float32.
__device__ __forceinline__ float at(const Row& r, unsigned int e) {
#if DT == 0
    return ((const float*)r.p)[e];
#elif DT == 2
    return __int_as_float(((unsigned int)((const unsigned short*)r.p)[e]) << 16);
#else
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(((const unsigned short*)r.p)[e]));
    return f;
#endif
}

__device__ __forceinline__ Row token_row(const void* logits, bool valid, unsigned long long t,
                                         unsigned int num_experts) {
    Row r = { (const char*)logits + (valid ? t : 0) * num_experts * (DT == 0 ? 4ull : 2ull) };
    return r;
}

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

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Rounds a float32 to the logits' precision and back: what torch does when
// it applies `sigmoid` to a float16 / bfloat16 tensor (computed in float32,
// stored in the tensor's dtype). Routers that call sigmoid on half-precision
// logits (LFM2-MoE) select on those rounded scores; matching it exactly is
// what keeps the logged choice identical to the model's.
__device__ __forceinline__ float round_to(float x) {
#if DT == 1
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    asm("cvt.f32.f16 %0, %1;" : "=f"(x) : "h"(h));
#elif DT == 2
    unsigned int b = __float_as_int(x);
    b += 0x7fffu + ((b >> 16) & 1u);  // round to nearest even (finite inputs)
    x = __int_as_float(b & 0xffff0000u);
#endif
    return x;
}

// Sigmoid score of expert e, in the logits' precision.
__device__ __forceinline__ float score(const Row& row, unsigned int e) {
    return round_to(sigmoid(at(row, e)));
}

// Sigmoid-mode score used for *selection*: score + bias[e].
__device__ __forceinline__ float biased_score(const Row& row, unsigned int e, const GateParams& gp) {
    if (!gp.bias) return score(row, e);
    float c = score(row, e) + gp.bias[e];
    return gp.round_biased ? round_to(c) : c;
}

// What top-k ranks for expert e: the raw logit in the softmax modes; in
// sigmoid mode the biased score, or -inf if e's group was not kept.
__device__ __forceinline__ float choice_key(const Row& row, unsigned int e, const GateParams& gp,
                                            unsigned int kept, unsigned int group_size) {
    if (!SIG) return rank_key(at(row, e));
    if (!((kept >> (e / group_size)) & 1u)) return neg_inf();
    return rank_key(biased_score(row, e, gp));
}

// Group-limited routing (sigmoid mode, num_groups > 1): bitmask of the
// top_groups groups, each scored by the sum of its two best biased scores
// (ties -> lower group index), same as the CPU router. The group's lanes
// share the work: each keeps its own top two of the group, then a butterfly
// merges them. Every lane of the warp must call this (shuffles).
__device__ unsigned int kept_groups(const Row& row, bool valid, unsigned int num_experts,
                                    const GateParams& gp, unsigned int lane, unsigned int lanes) {
    if (!SIG || gp.num_groups <= 1) return ALL_GROUPS;
    unsigned int size = num_experts / gp.num_groups;
    float group_score[MAX_GROUPS];
    for (unsigned int g = 0; g < gp.num_groups; ++g) {
        float a = neg_inf(), b = neg_inf();  // this lane's best two, a >= b
        if (valid) {
            for (unsigned int e = g * size + lane; e < (g + 1) * size; e += lanes) {
                float c = rank_key(biased_score(row, e, gp));
                if (c > a) { b = a; a = c; } else if (c > b) { b = c; }
            }
        }
        for (unsigned int off = lanes >> 1; off > 0; off >>= 1) {
            float oa = __shfl_xor_sync(FULL_MASK, a, off, lanes);
            float ob = __shfl_xor_sync(FULL_MASK, b, off, lanes);
            float hi = fmaxf(a, oa);
            b = fmaxf(fminf(a, oa), fmaxf(b, ob));
            a = hi;
        }
        group_score[g] = a + b;
    }
    unsigned int mask = 0;
    for (unsigned int r = 0; r < gp.top_groups; ++r) {
        int best = -1;
        for (unsigned int g = 0; g < gp.num_groups; ++g) {
            if ((mask >> g) & 1u) continue;
            if (best < 0 || group_score[g] > group_score[best]) best = (int)g;
        }
        mask |= 1u << best;
    }
    return mask;
}

// One lane's best expert among its stripe (lane, lane+G, lane+2G, ...),
// restricted (when has_prev) to experts ranked strictly after (pv, pi).
__device__ void lane_best(const Row& row, unsigned int num_experts, const GateParams& gp,
                          unsigned int kept, unsigned int group_size, unsigned int lane,
                          unsigned int lanes, bool has_prev, float pv, unsigned int pi,
                          float* bv, unsigned int* bi) {
    float v = neg_inf();
    unsigned int i = NO_EXPERT;
    for (unsigned int e = lane; e < num_experts; e += lanes) {
        float x = choice_key(row, e, gp, kept, group_size);
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
__device__ void group_top_k(const Row& row, bool valid, unsigned int num_experts,
                            unsigned int k, const GateParams& gp, unsigned int kept,
                            unsigned int lane, unsigned int lanes,
                            unsigned int* ids, float* vals) {
    unsigned int group_size = gp.num_groups > 1 ? num_experts / gp.num_groups : num_experts;
    float bv;
    unsigned int bi;
    if (valid) lane_best(row, num_experts, gp, kept, group_size, lane, lanes, false, 0.0f, 0, &bv, &bi);
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
        if (valid && bi == i)
            lane_best(row, num_experts, gp, kept, group_size, lane, lanes, true, v, i, &bv, &bi);
    }
}

// Softmax denominator for the token (mode values match GateMode):
//   0 = SoftmaxThenTopK: sum over all E logits, shared across the group.
//   1 = TopKThenSoftmax: sum over the k selected logits only (Mixtral).
//   2 = Sigmoid: unused (returns 0).
// vals[0] is the row max in both cases, which keeps exp() from overflowing.
__device__ float gate_denominator(const Row& row, bool valid, unsigned int num_experts,
                                  unsigned int k, int mode, unsigned int lane,
                                  unsigned int lanes, const float* vals) {
    float max = vals[0];
    float sum = 0.0f;
    if (mode == 0) {
        if (valid)
            for (unsigned int e = lane; e < num_experts; e += lanes) sum += expf(rank_key(at(row, e)) - max);
        for (unsigned int off = lanes >> 1; off > 0; off >>= 1)
            sum += __shfl_xor_sync(FULL_MASK, sum, off, lanes);
    } else if (mode == 1) {
        for (unsigned int j = 0; j < k; ++j) sum += expf(vals[j] - max);
    }
    return sum;
}

// Gate weights of the slots this lane writes (j = lane, lane + lanes, ...):
// mode-specific weights, then optional normalization by the sum over all k
// (sum + 1e-20, as in Hugging Face's norm_topk_prob; the sum is a group
// reduction of per-lane partial sums), then the scale. Every lane of the
// warp must call this (reductions).
__device__ void compute_gates(const Row& row, bool valid, unsigned int num_experts,
                              unsigned int k, const GateParams& gp, unsigned int lane,
                              unsigned int lanes, const unsigned int* ids, const float* vals,
                              float* gates) {
    float sum = gate_denominator(row, valid, num_experts, k, gp.mode, lane, lanes, vals);
    float part = 0.0f;
    if (valid) {
        for (unsigned int j = lane; j < k; j += lanes) {
            gates[j] = SIG ? score(row, ids[j]) : expf(vals[j] - vals[0]) / sum;
            part += gates[j];
        }
    }
    float scale = gp.scale;
    if (gp.normalize) {
        for (unsigned int off = lanes >> 1; off > 0; off >>= 1)
            part += __shfl_xor_sync(FULL_MASK, part, off, lanes);
        scale /= part + 1e-20f;
    }
    if (valid)
        for (unsigned int j = lane; j < k; j += lanes) gates[j] *= scale;
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
extern "C" __global__ void moe_route(const void* logits, unsigned int num_tokens,
                                     unsigned int num_experts, unsigned int k, GateParams gp,
                                     RouteEntry* out, unsigned int lanes) {
    unsigned long long t;
    unsigned int lane;
    token_and_lane(lanes, &t, &lane);
    bool valid = t < num_tokens;
    Row row = token_row(logits, valid, t, num_experts);
    unsigned int ids[MAX_K];
    float vals[MAX_K], gates[MAX_K];
    unsigned int kept = kept_groups(row, valid, num_experts, gp, lane, lanes);
    group_top_k(row, valid, num_experts, k, gp, kept, lane, lanes, ids, vals);
    compute_gates(row, valid, num_experts, k, gp, lane, lanes, ids, vals, gates);
    if (!valid) return;
    for (unsigned int j = lane; j < k; j += lanes) {
        out[t * k + j].expert_id = ids[j];
        out[t * k + j].gate = gates[j];
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
        const void* logits, unsigned int num_tokens, unsigned int num_experts,
        unsigned int k, GateParams gp, unsigned int layer_id, unsigned int token_offset,
        RouteEntry* routes, unsigned int* expert_counts, unsigned int* gathered,
        TelemetryRecord* log, unsigned long long capacity,
        unsigned long long* log_counters, int write_dispatch, unsigned int lanes) {
    unsigned long long t;
    unsigned int lane;
    token_and_lane(lanes, &t, &lane);
    bool valid = t < num_tokens;
    Row row = token_row(logits, valid, t, num_experts);
    unsigned int ids[MAX_K];
    float vals[MAX_K], gates[MAX_K];
    unsigned int kept = kept_groups(row, valid, num_experts, gp, lane, lanes);
    group_top_k(row, valid, num_experts, k, gp, kept, lane, lanes, ids, vals);
    compute_gates(row, valid, num_experts, k, gp, lane, lanes, ids, vals, gates);

    unsigned long long base = 0;
    if (valid && lane == 0) base = atomicAdd(&log_counters[0], (unsigned long long)k);
    base = __shfl_sync(FULL_MASK, base, 0, lanes);
    if (!valid) return;

    unsigned long long token_base = gp.token_base ? *gp.token_base : 0ull;
    unsigned int dropped = 0;
    for (unsigned int j = lane; j < k; j += lanes) {
        unsigned int e = ids[j];
        float gate = gates[j];
        if (write_dispatch) {
            unsigned int a = (unsigned int)t * k + j;
            routes[a].expert_id = e;
            routes[a].gate = gate;
            unsigned int slot = atomicAdd(&expert_counts[e], 1u);
            gathered[e * num_tokens + slot] = a;
        }

        unsigned long long idx = base + j;
        if (idx < capacity) {
            log[idx].token_id = (unsigned int)(token_base + token_offset + t);
            log[idx].layer_id = (unsigned short)layer_id;
            log[idx].expert_id = (unsigned short)e;
            log[idx].gate = gate;
        } else {
            ++dropped;
        }
    }
    if (dropped > 0) atomicAdd(&log_counters[1], (unsigned long long)dropped);
}

// Moves the device-side token counter past one forward pass. Launched on
// the model's stream after the pass's last MoE layer, so inside a captured
// CUDA graph it replays too and every replay logs new token ids.
extern "C" __global__ void moe_advance_tokens(unsigned long long* token_base, unsigned long long n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *token_base += n;
}
"#;

/// Number of compiled kernel variants: 3 logits dtypes x (softmax, sigmoid).
pub const NUM_VARIANTS: usize = 6;

/// Variant index for a logits dtype code (0 f32, 1 f16, 2 bf16) and mode.
pub fn variant(dtype_code: usize, sigmoid: bool) -> usize {
    dtype_code * 2 + sigmoid as usize
}

/// Kernel source for one variant, with constants substituted, ready for NVRTC.
pub fn source(variant: usize) -> String {
    assert!(variant < NUM_VARIANTS);
    SRC.replace("MOESCOPE_MAX_K", &MAX_K.to_string())
        .replace(
            "MOESCOPE_MAX_GROUPS",
            &crate::router::MAX_GROUPS.to_string(),
        )
        .replace("MOESCOPE_DT", &(variant / 2).to_string())
        .replace("MOESCOPE_SIG", &(variant % 2).to_string())
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
        let picks: Vec<_> = [1, 8, 32, 40, 64, 128, 256]
            .iter()
            .map(|&e| lanes_per_token(e))
            .collect();
        assert_eq!(picks, [1, 2, 8, 16, 16, 32, 32]);
    }
}
