//! CUDA C source for the MoE kernels, compiled at runtime by NVRTC.
//!
//! The source deliberately avoids every header (NVRTC has no libc), so it
//! uses `unsigned int` etc. directly and builds -inf from its bit pattern.
//! The struct layouts here must match the `#[repr(C)]` Rust types:
//! `RouteEntry` = [`crate::router::Route`],
//! `TelemetryRecord` = [`crate::telemetry::TelemetryRecord`].

/// Largest `k` the kernels support. Each thread keeps its running top-k in
/// registers / local memory, so this is a fixed-size array bound.
pub const MAX_K: usize = 32;

/// Kernel entry-point names, kept next to the source that defines them.
pub const ROUTE_KERNEL: &str = "moe_route";
pub const DISPATCH_LOG_KERNEL: &str = "moe_route_dispatch_log";

/// Kernel source. `MOESCOPE_MAX_K` is substituted from [`MAX_K`] by [`source`].
const SRC: &str = r#"
#define MAX_K MOESCOPE_MAX_K

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

// Top-k selection for one token over its E logits.
//
// Single pass over experts in index order, maintaining a sorted (descending)
// list of the best k seen so far. A new expert displaces an entry only if it
// is *strictly* greater, so on ties the lower expert index stays ahead: the
// same order the CPU sort produces. E is small (8-128), so O(E*k) is fine and
// avoids any shared memory or warp cooperation.
__device__ void select_top_k(const float* row, unsigned int num_experts, unsigned int k,
                             unsigned int* ids, float* vals) {
    unsigned int n = 0;
    for (unsigned int e = 0; e < num_experts; ++e) {
        float x = rank_key(row[e]);
        if (n == k && !(x > vals[k - 1])) continue;   // not good enough
        unsigned int p = (n < k) ? n++ : k - 1;      // append, or evict the worst
        while (p > 0 && x > vals[p - 1]) {             // bubble up past smaller ones
            vals[p] = vals[p - 1];
            ids[p] = ids[p - 1];
            --p;
        }
        vals[p] = x;
        ids[p] = e;
    }
}

// Gate weights for the selected experts (mode values match GateMode):
//   0 = SoftmaxThenTopK: softmax over all E logits, keep the selected probs.
//   1 = TopKThenSoftmax: softmax over the k selected logits only (Mixtral).
// vals[0] is the row max in both cases, which keeps exp() from overflowing.
__device__ void compute_gates(const float* row, unsigned int num_experts, unsigned int k,
                              int mode, const float* vals, float* gates) {
    float max = vals[0];
    float sum = 0.0f;
    if (mode == 0) {
        for (unsigned int e = 0; e < num_experts; ++e) sum += expf(rank_key(row[e]) - max);
    } else {
        for (unsigned int j = 0; j < k; ++j) sum += expf(vals[j] - max);
    }
    for (unsigned int j = 0; j < k; ++j) gates[j] = expf(vals[j] - max) / sum;
}

// Full routing decision for token t into registers.
__device__ void route_one_token(const float* logits, unsigned int t, unsigned int num_experts,
                                unsigned int k, int mode, unsigned int* ids, float* gates) {
    float vals[MAX_K];
    const float* row = logits + (unsigned long long)t * num_experts;
    select_top_k(row, num_experts, k, ids, vals);
    compute_gates(row, num_experts, k, mode, vals, gates);
}

// Phase 3: one thread per token. Writes out[t * k + j] (best expert first).
extern "C" __global__ void moe_route(const float* logits, unsigned int num_tokens,
                                     unsigned int num_experts, unsigned int k, int mode,
                                     RouteEntry* out) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    unsigned int ids[MAX_K];
    float gates[MAX_K];
    route_one_token(logits, t, num_experts, k, mode, ids, gates);
    for (unsigned int j = 0; j < k; ++j) {
        out[t * k + j].expert_id = ids[j];
        out[t * k + j].gate = gates[j];
    }
}

// Phase 4: route + dispatch + telemetry in one pass, one thread per token.
//
// Dispatch: atomicAdd on expert_counts[e] hands the token a unique slot in
// expert e's block of `gathered` (stride num_tokens, which can never
// overflow since a token picks each expert at most once). The final
// expert_counts is the expert load histogram.
//
// Telemetry: each token reserves k consecutive records with ONE atomicAdd on
// log_counters[0] (the append cursor, which keeps counting past capacity).
// Records that land at or beyond `capacity` are not written; they are counted
// in log_counters[1] (dropped). Nothing is copied to the host here.
//
// write_dispatch == 0 is the telemetry-only mode: routes, expert_counts and
// gathered are never touched (they may be null), so logging costs one launch
// and no allocations.
extern "C" __global__ void moe_route_dispatch_log(
        const float* logits, unsigned int num_tokens, unsigned int num_experts,
        unsigned int k, int mode, unsigned int layer_id, unsigned int token_offset,
        RouteEntry* routes, unsigned int* expert_counts, unsigned int* gathered,
        TelemetryRecord* log, unsigned long long capacity,
        unsigned long long* log_counters, int write_dispatch) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    unsigned int ids[MAX_K];
    float gates[MAX_K];
    route_one_token(logits, t, num_experts, k, mode, ids, gates);

    unsigned long long base = atomicAdd(&log_counters[0], (unsigned long long)k);
    unsigned int dropped = 0;
    for (unsigned int j = 0; j < k; ++j) {
        unsigned int e = ids[j];
        if (write_dispatch) {
            unsigned int a = t * k + j;
            routes[a].expert_id = e;
            routes[a].gate = gates[j];
            unsigned int slot = atomicAdd(&expert_counts[e], 1u);
            gathered[e * num_tokens + slot] = a;
        }

        unsigned long long idx = base + j;
        if (idx < capacity) {
            log[idx].token_id = token_offset + t;
            log[idx].layer_id = (unsigned short)layer_id;
            log[idx].expert_id = (unsigned short)e;
            log[idx].gate = gates[j];
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
