//! The backend trait: the line between engine code (written once) and device
//! code (one implementation per device). ADR 0003.
//!
//! - **Op-level.** The Llama model (`crate::model`) is a sequence of calls to
//!   the ops below. Each op is a future CUDA kernel, and the CPU backend is its
//!   reference implementation.
//! - **Activations live in one workspace** the backend allocates at startup.
//!   The engine carves it up with a bump arena (`crate::arena`) and names
//!   pieces of it with [`Region`]s. Ops never allocate.
//! - **KV bytes belong to the backend** ([`Backend::KvPool`]). Which slot holds
//!   which token is the engine's bookkeeping, passed in as block tables and
//!   slot mappings ([`PrefillMeta`], [`DecodeMeta`]; see [`paged`]).
//! - **Prefill and decode attention are separate ops** (ADR 0004).
//! - **The only device→host reads** are [`Backend::argmax`] (the one sync per
//!   step) and [`Backend::read`] (debug and tests only, never in the hot loop).
//!
//! The op methods are `unsafe`: they take raw workspace regions and trust the
//! caller that an output never overlaps an input. `arena::Scope` is the only
//! caller and guarantees it; see the SAFETY comment there.

pub mod paged;

pub use crate::error::{EngineError, Result};
pub use paged::{
    BlockTableTooShort, DecodeMeta, DecodeSeq, KvLayout, PrefillMeta, PrefillSeq, slot_of,
};

use crate::weights::Weight;

/// A piece of the activation workspace holding a row-major `[rows, cols]` f32
/// matrix. Offset and sizes are in elements, not bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Region {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A compute device. See the module docs for the design.
///
/// # Safety contract for the `unsafe` op methods
///
/// The caller guarantees, for every op:
/// 1. every [`Region`] lies inside the workspace from [`Backend::alloc_workspace`];
/// 2. an output region overlaps no input region (an in-place op names its
///    in-out region once, and its other inputs don't overlap it);
/// 3. shapes are consistent (checked by `arena::Scope` before calling).
///
/// The backend may then hand out `&mut` views of output regions alongside `&`
/// views of input regions without aliasing.
pub trait Backend {
    /// A weight matrix or vector, resident on the device.
    type Weight;
    /// The KV cache memory. See [`KvLayout`].
    type KvPool;

    fn name(&self) -> &'static str;

    /// Move a host weight onto the device. Matrices are `[out, in]` row-major.
    fn upload(&self, weight: Weight) -> Result<Self::Weight>;
    /// Shape of an uploaded weight.
    fn weight_shape<'w>(&self, weight: &'w Self::Weight) -> &'w [usize];
    /// Allocate the KV pool. Called once at startup.
    fn alloc_kv_pool(&self, layout: KvLayout) -> Result<Self::KvPool>;
    fn kv_layout(&self, pool: &Self::KvPool) -> KvLayout;
    /// Allocate the activation workspace (`elems` f32). Called once at startup.
    fn alloc_workspace(&mut self, elems: usize) -> Result<()>;
    fn workspace_elems(&self) -> usize;

    /// `out[t] = table[tokens[t]]`. tokens `[T]`, out `[T, hidden]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn embed(&self, table: &Self::Weight, tokens: &[u32], out: Region) -> Result<()>;

    /// `out[t] = x[t] / sqrt(mean(x[t]²) + eps) * weight`.
    /// x, out `[T, H]`, weight `[H]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn rmsnorm(&self, x: Region, weight: &Self::Weight, eps: f32, out: Region)
    -> Result<()>;

    /// `out = x · weightᵀ`. x `[T, in]`, weight `[out, in]`, out `[T, out]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn matmul(&self, x: Region, weight: &Self::Weight, out: Region) -> Result<()>;

    /// Rotary position embedding, in place. x `[T, heads * head_dim]`,
    /// positions `[T]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn rope(&self, x: Region, positions: &[u32], head_dim: usize) -> Result<()>;

    /// Store row `t`'s K and V at `slots[t]` of `layer`.
    /// keys, values `[T, kv_heads * head_dim]`, slots `[T]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn write_kv(
        &self,
        pool: &mut Self::KvPool,
        layer: usize,
        keys: Region,
        values: Region,
        slots: &[u32],
    ) -> Result<()>;

    /// Attention for a prefill step. Each row attends to `row_num_keys[t]`
    /// keys of its sequence, read from the pool through that sequence's block
    /// table, so tokens cached by earlier steps are included.
    /// queries, out `[T, heads * head_dim]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn prefill_attention(
        &self,
        queries: Region,
        pool: &Self::KvPool,
        layer: usize,
        meta: &PrefillMeta,
        out: Region,
    ) -> Result<()>;

    /// Attention for a decode step: one row per sequence.
    /// queries, out `[S, heads * head_dim]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn decode_attention(
        &self,
        queries: Region,
        pool: &Self::KvPool,
        layer: usize,
        meta: &DecodeMeta,
        out: Region,
    ) -> Result<()>;

    /// `gate = silu(gate) * up`, in place. gate, up `[T, intermediate]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn silu_mul(&self, gate: Region, up: Region) -> Result<()>;

    /// `sum += x`, in place. sum, x `[T, H]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn add(&self, sum: Region, x: Region) -> Result<()>;

    /// `out[i] = x[rows[i]]`. x `[T, H]`, rows `[S]`, out `[S, H]`.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn gather_rows(&self, x: Region, rows: &[u32], out: Region) -> Result<()>;

    /// Index of the largest value in each row, lowest index on ties (matches
    /// `torch.argmax`). x `[S, V]` → `[S]`.
    /// **The one device→host sync per step.**
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn argmax(&self, x: Region) -> Result<Vec<u32>>;

    /// Copy a region to the host. Debug and tests only; never in the hot loop.
    ///
    /// # Safety
    /// See the trait-level contract.
    unsafe fn read(&self, x: Region) -> Result<Vec<f32>>;
}
