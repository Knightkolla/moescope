//! Routing telemetry: one fixed-size record per (token, expert) decision.
//!
//! The point of logging inside the routing kernel is that nothing crosses
//! PCIe per step: records accumulate in a pre-allocated buffer and are
//! drained to host once, at the end of a run or an eval window. This module
//! defines the record layout shared with the GPU and a host-side log with
//! the same overflow semantics, used for the CPU path and as the test oracle.

use std::fmt;

use crate::router::Route;

/// One routing decision: token `token_id` in MoE layer `layer_id` was sent to
/// expert `expert_id` with gate weight `gate`. `flags` is a bit set, see
/// [`CAPACITY_DROP`].
///
/// `#[repr(C)]`, 12 bytes, no padding; identical to `TelemetryRecord` in the
/// CUDA source and to the numpy dtype exposed to Python
/// (`<u4 token_id, u1 layer_id, u1 flags, <u2 expert_id, <f4 gate`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TelemetryRecord {
    pub token_id: u32,
    pub layer_id: u8,
    pub flags: u8,
    pub expert_id: u16,
    pub gate: f32,
}

/// `flags` bit: the router chose this expert but the expert never computed
/// the token, because the expert was full (capacity-factor MoE, e.g.
/// Switch Transformer). `gate` is still the router's gate. Set only when the
/// caller passes the model's own dispatch mask ([`KeptMask`]).
pub const CAPACITY_DROP: u8 = 1;

/// What the experts actually received: `mask[t * num_experts + e] != 0` iff
/// expert `e` computed token `t` (the model's dispatch mask after capacity).
#[derive(Debug, Clone, Copy)]
pub struct KeptMask<'a> {
    pub mask: &'a [u8],
    pub num_experts: usize,
}

const _: () = assert!(std::mem::size_of::<TelemetryRecord>() == 12);
const _: () = assert!(std::mem::align_of::<TelemetryRecord>() == 4);

/// The contents of a log at drain time.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Drained {
    /// Records that fit in the buffer. On the GPU their order is the order
    /// threads won the append atomic, not token order.
    pub records: Vec<TelemetryRecord>,
    /// Records that arrived after the buffer was full and were discarded.
    /// Non-zero means the capacity was too small for the logging window.
    pub dropped: u64,
}

/// Reasons a batch cannot be represented in [`TelemetryRecord`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryError {
    /// `expert_id` is stored as u16, so layers with > 65536 experts can't be logged.
    ExpertIdOverflow { expert_id: u32 },
    /// `token_offset + num_tokens` must fit in the u32 `token_id`.
    TokenIdOverflow {
        token_offset: u32,
        num_tokens: usize,
    },
    /// A [`KeptMask`] must be `num_tokens * num_experts` bytes.
    KeptMaskShape { expected: usize, found: usize },
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TelemetryError::ExpertIdOverflow { expert_id } => {
                write!(f, "expert_id {expert_id} does not fit in u16")
            }
            TelemetryError::TokenIdOverflow {
                token_offset,
                num_tokens,
            } => {
                write!(
                    f,
                    "token_offset {token_offset} + {num_tokens} tokens overflows u32"
                )
            }
            TelemetryError::KeptMaskShape { expected, found } => {
                write!(
                    f,
                    "kept mask has {found} entries, expected {expected} (tokens x experts)"
                )
            }
        }
    }
}

impl std::error::Error for TelemetryError {}

/// Checks that token ids `token_offset .. token_offset + num_tokens` fit in u32.
pub fn check_token_range(token_offset: u32, num_tokens: usize) -> Result<(), TelemetryError> {
    let end = token_offset as u64 + num_tokens as u64;
    if end > u32::MAX as u64 + 1 {
        return Err(TelemetryError::TokenIdOverflow {
            token_offset,
            num_tokens,
        });
    }
    Ok(())
}

/// Telemetry records for one routed batch, in `[token * k + j]` order.
/// Token ids start at `token_offset`, so successive batches can be logged
/// into one buffer without colliding. Panics if `k == 0`.
pub fn records_for(
    routes: &[Route],
    k: usize,
    layer_id: u8,
    token_offset: u32,
) -> Result<Vec<TelemetryRecord>, TelemetryError> {
    records_for_masked(routes, k, layer_id, token_offset, None)
}

/// [`records_for`], flagging [`CAPACITY_DROP`] on every decision the
/// model's dispatch mask says was never computed.
pub fn records_for_masked(
    routes: &[Route],
    k: usize,
    layer_id: u8,
    token_offset: u32,
    kept: Option<KeptMask<'_>>,
) -> Result<Vec<TelemetryRecord>, TelemetryError> {
    assert!(k > 0, "k must be >= 1");
    let n = routes.len() / k;
    check_token_range(token_offset, n)?;
    if let Some(m) = kept {
        let expected = n * m.num_experts;
        if m.mask.len() != expected {
            return Err(TelemetryError::KeptMaskShape {
                expected,
                found: m.mask.len(),
            });
        }
    }
    routes
        .iter()
        .enumerate()
        .map(|(a, r)| {
            let expert_id =
                u16::try_from(r.expert_id).map_err(|_| TelemetryError::ExpertIdOverflow {
                    expert_id: r.expert_id,
                })?;
            let t = a / k;
            let dropped = kept.is_some_and(|m| {
                m.mask
                    .get(t * m.num_experts + r.expert_id as usize)
                    .is_some_and(|&x| x == 0)
            });
            Ok(TelemetryRecord {
                token_id: token_offset + t as u32,
                layer_id,
                flags: if dropped { CAPACITY_DROP } else { 0 },
                expert_id,
                gate: r.gate,
            })
        })
        .collect()
}

/// Bounded host-side telemetry log, the CPU twin of `gpu::TelemetryBuffer`.
///
/// Capacity is fixed up front (as on the GPU, where the buffer cannot grow
/// mid-kernel); once full, further records are counted in `dropped` rather
/// than silently lost or reallocating.
#[derive(Debug, Clone)]
pub struct HostTelemetry {
    records: Vec<TelemetryRecord>,
    capacity: usize,
    dropped: u64,
}

impl HostTelemetry {
    /// A log that holds at most `capacity` records between drains.
    pub fn new(capacity: usize) -> Self {
        HostTelemetry {
            records: Vec::with_capacity(capacity),
            capacity,
            dropped: 0,
        }
    }

    /// Maximum records held between drains.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Records currently held.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if nothing has been logged since the last drain.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty() && self.dropped == 0
    }

    /// Appends every decision of a routed batch, dropping (and counting)
    /// whatever does not fit.
    pub fn log_routes(
        &mut self,
        routes: &[Route],
        k: usize,
        layer_id: u8,
        token_offset: u32,
    ) -> Result<(), TelemetryError> {
        self.log_routes_masked(routes, k, layer_id, token_offset, None)
    }

    /// [`log_routes`](Self::log_routes) with the model's dispatch mask, so
    /// decisions the experts never computed carry [`CAPACITY_DROP`].
    pub fn log_routes_masked(
        &mut self,
        routes: &[Route],
        k: usize,
        layer_id: u8,
        token_offset: u32,
        kept: Option<KeptMask<'_>>,
    ) -> Result<(), TelemetryError> {
        let new = records_for_masked(routes, k, layer_id, token_offset, kept)?;
        let room = self.capacity - self.records.len();
        let kept = new.len().min(room);
        self.dropped += (new.len() - kept) as u64;
        self.records.extend_from_slice(&new[..kept]);
        Ok(())
    }

    /// Hands over everything logged so far and resets the log.
    pub fn drain(&mut self) -> Drained {
        Drained {
            records: std::mem::take(&mut self.records),
            dropped: std::mem::take(&mut self.dropped),
        }
    }
}
