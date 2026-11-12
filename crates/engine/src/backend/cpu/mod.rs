//! CPU backend: f32, host memory, rayon for parallelism (ADR 0001).
//!
//! It is the numerical reference for every future CUDA kernel, so each op is
//! the plain textbook loop. The only performance work is the obvious kind:
//! run independent outputs in parallel, and write the dot product so the
//! compiler can vectorize it. It is not a performance target.

mod attention;
pub mod rope;

use std::cell::UnsafeCell;

use rayon::prelude::*;

use super::{Backend, DecodeMeta, EngineError, KvLayout, PrefillMeta, Region, Result};
use crate::config::ModelConfig;
use crate::weights::Weight;
use attention::{AttentionInputs, check_block_tables, paged_attention};
use rope::RopeTable;

/// Weight rows per rayon task in `matmul`. 16 rows × 2048 inputs × 4 B =
/// 128 KB, which fits the 7700X's 1 MB L2 cache alongside the activations.
/// TODO(Kamdyn): sweep 4–64 if CPU speed ever matters (it doesn't yet).
const MATMUL_ROWS_PER_TASK: usize = 16;

/// Independent partial sums in `dot`. f32 addition isn't associative, so the
/// compiler won't vectorize one running sum; 16 separate sums fill one 512-bit
/// AVX-512 register (16 × f32).
const DOT_LANES: usize = 16;

pub struct CpuKvPool {
    layout: KvLayout,
    /// `[layers][num_slots * kv_heads * head_dim]`
    keys: Vec<Vec<f32>>,
    /// `[layers][num_slots * kv_heads * head_dim]`
    values: Vec<Vec<f32>>,
}

pub struct CpuBackend {
    /// The activation workspace. `UnsafeCell` because ops hand out `&` views of
    /// input regions and `&mut` views of output regions at the same time,
    /// through `&self`. That they don't overlap is the caller's contract (see
    /// `Backend`).
    workspace: Box<[UnsafeCell<f32>]>,
    rope: RopeTable,
}

impl CpuBackend {
    /// `max_len`: positions the RoPE table covers (the engine's max model length).
    pub fn new(config: &ModelConfig, max_len: usize) -> Self {
        CpuBackend {
            workspace: Box::new([]),
            rope: RopeTable::new(config, max_len),
        }
    }

    fn check_in_workspace(&self, region: Region) -> Result<()> {
        if region.offset + region.len() > self.workspace.len() {
            return Err(EngineError::OutsideWorkspace);
        }
        Ok(())
    }

    /// Shared view of a workspace region.
    ///
    /// # Safety
    /// No `&mut` view of an overlapping region may be alive (the trait
    /// contract: inputs never overlap outputs). Bounds are checked here.
    unsafe fn view(&self, region: Region) -> Result<&[f32]> {
        self.check_in_workspace(region)?;
        let start = self.workspace.as_ptr().wrapping_add(region.offset);
        // SAFETY: in bounds (checked above); `UnsafeCell<f32>` has the same
        // layout as `f32`; no overlapping `&mut` exists, per the contract.
        Ok(unsafe { std::slice::from_raw_parts(UnsafeCell::raw_get(start), region.len()) })
    }

    /// Exclusive view of a workspace region.
    ///
    /// # Safety
    /// No other view of an overlapping region may be alive (the trait
    /// contract: outputs overlap nothing). Bounds are checked here.
    #[allow(clippy::mut_from_ref)]
    unsafe fn view_mut(&self, region: Region) -> Result<&mut [f32]> {
        self.check_in_workspace(region)?;
        let start = self.workspace.as_ptr().wrapping_add(region.offset);
        // SAFETY: in bounds (checked above); writing through a shared reference
        // is allowed because the elements are `UnsafeCell`; exclusivity is the
        // contract.
        Ok(unsafe { std::slice::from_raw_parts_mut(UnsafeCell::raw_get(start), region.len()) })
    }
}

/// `Σ a[i] · b[i]`, using `DOT_LANES` independent partial sums.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_rest) = a.as_chunks::<DOT_LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<DOT_LANES>();
    let mut partial = [0.0f32; DOT_LANES];
    for (a_chunk, b_chunk) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..DOT_LANES {
            partial[lane] += a_chunk[lane] * b_chunk[lane];
        }
    }
    let mut sum: f32 = partial.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        sum += x * y;
    }
    sum
}

/// A raw output pointer that rayon tasks may share. Used only in `matmul`,
/// where each task writes a disjoint set of elements that isn't one contiguous
/// chunk (a column range of every output row).
#[derive(Clone, Copy)]
struct SharedOutput(*mut f32);

// SAFETY: every task writes disjoint elements (see `matmul`), and nothing
// reads them until rayon's parallel loop has finished.
unsafe impl Send for SharedOutput {}
// SAFETY: as above.
unsafe impl Sync for SharedOutput {}

impl SharedOutput {
    /// Pointer to element `index`. A method rather than `.0` so closures
    /// capture the whole (Sync) struct, not the bare (non-Sync) pointer field.
    fn at(self, index: usize) -> *mut f32 {
        self.0.wrapping_add(index)
    }
}

impl Backend for CpuBackend {
    type Weight = Weight;
    type KvPool = CpuKvPool;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn upload(&self, weight: Weight) -> Result<Weight> {
        Ok(weight) // already in host memory
    }

    fn weight_shape<'w>(&self, weight: &'w Weight) -> &'w [usize] {
        weight.shape()
    }

    fn alloc_kv_pool(&self, layout: KvLayout) -> Result<CpuKvPool> {
        // `vec![0.0; n]` uses calloc: the OS hands out zeroed pages lazily, so
        // a 2 GiB pool costs nothing until it's written.
        let elems = layout.layer_elems();
        Ok(CpuKvPool {
            layout,
            keys: (0..layout.num_layers).map(|_| vec![0.0; elems]).collect(),
            values: (0..layout.num_layers).map(|_| vec![0.0; elems]).collect(),
        })
    }

    fn kv_layout(&self, pool: &CpuKvPool) -> KvLayout {
        pool.layout
    }

    fn alloc_workspace(&mut self, elems: usize) -> Result<()> {
        self.workspace = (0..elems).map(|_| UnsafeCell::new(0.0)).collect();
        Ok(())
    }

    fn workspace_elems(&self) -> usize {
        self.workspace.len()
    }

    unsafe fn embed(&self, table: &Weight, tokens: &[u32], out: Region) -> Result<()> {
        // SAFETY: the caller upholds the `Backend` contract.
        let out = unsafe { self.view_mut(out)? };
        let hidden = table.shape()[1];
        for (out_row, &token) in out.chunks_exact_mut(hidden).zip(tokens) {
            out_row.copy_from_slice(&table.data()[token as usize * hidden..][..hidden]);
        }
        Ok(())
    }

    unsafe fn rmsnorm(&self, x: Region, weight: &Weight, eps: f32, out: Region) -> Result<()> {
        // SAFETY: the caller upholds the `Backend` contract.
        let (x, out) = unsafe { (self.view(x)?, self.view_mut(out)?) };
        let hidden = weight.numel();
        out.par_chunks_mut(hidden)
            .zip(x.par_chunks(hidden))
            .for_each(|(out_row, x_row)| {
                let mean_square = dot(x_row, x_row) / hidden as f32;
                let inv_rms = 1.0 / (mean_square + eps).sqrt();
                for ((out, &x), &w) in out_row.iter_mut().zip(x_row).zip(weight.data()) {
                    // Same order as HF: weight * (x * rsqrt(mean_square + eps)).
                    *out = w * (x * inv_rms);
                }
            });
        Ok(())
    }

    unsafe fn matmul(&self, x: Region, weight: &Weight, out: Region) -> Result<()> {
        // SAFETY: the caller upholds the `Backend` contract.
        let (x_data, out_data) = unsafe { (self.view(x)?, self.view_mut(out)?) };
        let out_features = weight.shape()[0];
        let in_features = weight.shape()[1];
        let input_rows = x.rows;
        let output = SharedOutput(out_data.as_mut_ptr());
        // Parallel over chunks of weight rows (= output columns). Each weight
        // row is read from memory once and reused for every input row while
        // it's in cache. At decode (one input row) this streams the weights
        // once: the memory-bound case.
        weight
            .data()
            .par_chunks(MATMUL_ROWS_PER_TASK * in_features)
            .enumerate()
            .for_each(|(chunk_index, weight_rows)| {
                for (i, weight_row) in weight_rows.chunks_exact(in_features).enumerate() {
                    let out_col = chunk_index * MATMUL_ROWS_PER_TASK + i;
                    for row in 0..input_rows {
                        let x_row = &x_data[row * in_features..][..in_features];
                        let value = dot(x_row, weight_row);
                        // SAFETY: `out_col` belongs to this task alone (weight
                        // row chunks don't overlap), so element [row, out_col]
                        // is written by exactly one task. row < input_rows and
                        // out_col < out_features keep it inside `out_data`.
                        unsafe { *output.at(row * out_features + out_col) = value };
                    }
                }
            });
        Ok(())
    }

    unsafe fn rope(&self, x: Region, positions: &[u32], head_dim: usize) -> Result<()> {
        if head_dim != 2 * self.rope.half {
            return Err(EngineError::Shape {
                op: "rope",
                msg: format!(
                    "head_dim {head_dim}, table built for {}",
                    2 * self.rope.half
                ),
            });
        }
        if let Some(&pos) = positions.iter().find(|&&p| p as usize >= self.rope.max_len) {
            return Err(EngineError::PositionOutOfRange {
                pos: pos as usize,
                max: self.rope.max_len,
            });
        }
        // SAFETY: the caller upholds the `Backend` contract; x is the in-out region.
        let x_data = unsafe { self.view_mut(x)? };
        let table = &self.rope; // borrow the table, not `self` (which isn't Sync)
        x_data
            .par_chunks_mut(x.cols)
            .zip(positions.par_iter())
            .for_each(|(row, &pos)| table.apply(row, pos as usize));
        Ok(())
    }

    unsafe fn write_kv(
        &self,
        pool: &mut CpuKvPool,
        layer: usize,
        keys: Region,
        values: Region,
        slots: &[u32],
    ) -> Result<()> {
        let num_slots = pool.layout.num_slots();
        if let Some(&slot) = slots.iter().find(|&&s| s as usize >= num_slots) {
            return Err(EngineError::SlotOutOfRange {
                op: "write_kv",
                slot: slot as usize,
                num_slots,
            });
        }
        let slot_width = keys.cols;
        // SAFETY: the caller upholds the `Backend` contract. The pool is
        // separate memory from the workspace.
        let (keys, values) = unsafe { (self.view(keys)?, self.view(values)?) };
        for (row, &slot) in slots.iter().enumerate() {
            let slot = slot as usize;
            let src = row * slot_width..(row + 1) * slot_width;
            let dst = slot * slot_width..(slot + 1) * slot_width;
            pool.keys[layer][dst.clone()].copy_from_slice(&keys[src.clone()]);
            pool.values[layer][dst].copy_from_slice(&values[src]);
        }
        Ok(())
    }

    unsafe fn prefill_attention(
        &self,
        queries: Region,
        pool: &CpuKvPool,
        layer: usize,
        meta: &PrefillMeta,
        out: Region,
    ) -> Result<()> {
        check_block_tables(
            "prefill_attention",
            &meta.block_tables,
            pool.layout.num_blocks,
        )?;
        let inputs = attention_inputs(
            pool,
            queries.cols,
            meta.block_size,
            &meta.block_tables,
            &meta.row_seq,
            &meta.row_num_keys,
        );
        // SAFETY: the caller upholds the `Backend` contract.
        let (queries, out) = unsafe { (self.view(queries)?, self.view_mut(out)?) };
        paged_attention(queries, out, pool, layer, &inputs);
        Ok(())
    }

    unsafe fn decode_attention(
        &self,
        queries: Region,
        pool: &CpuKvPool,
        layer: usize,
        meta: &DecodeMeta,
        out: Region,
    ) -> Result<()> {
        check_block_tables(
            "decode_attention",
            &meta.block_tables,
            pool.layout.num_blocks,
        )?;
        let inputs = attention_inputs(
            pool,
            queries.cols,
            meta.block_size,
            &meta.block_tables,
            &meta.row_seq,
            &meta.row_num_keys,
        );
        // SAFETY: the caller upholds the `Backend` contract.
        let (queries, out) = unsafe { (self.view(queries)?, self.view_mut(out)?) };
        paged_attention(queries, out, pool, layer, &inputs);
        Ok(())
    }

    unsafe fn silu_mul(&self, gate: Region, up: Region) -> Result<()> {
        // SAFETY: the caller upholds the `Backend` contract; gate is the in-out region.
        let (gate, up) = unsafe { (self.view_mut(gate)?, self.view(up)?) };
        gate.par_iter_mut().zip(up.par_iter()).for_each(|(g, &u)| {
            // silu(x) = x · sigmoid(x) = x / (1 + e^-x)
            *g = *g / (1.0 + (-*g).exp()) * u;
        });
        Ok(())
    }

    unsafe fn add(&self, sum: Region, x: Region) -> Result<()> {
        // SAFETY: the caller upholds the `Backend` contract; sum is the in-out region.
        let (sum, x) = unsafe { (self.view_mut(sum)?, self.view(x)?) };
        sum.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(s, &x)| *s += x);
        Ok(())
    }

    unsafe fn gather_rows(&self, x: Region, rows: &[u32], out: Region) -> Result<()> {
        let width = x.cols;
        // SAFETY: the caller upholds the `Backend` contract.
        let (x, out) = unsafe { (self.view(x)?, self.view_mut(out)?) };
        for (out_row, &row) in out.chunks_exact_mut(width).zip(rows) {
            out_row.copy_from_slice(&x[row as usize * width..][..width]);
        }
        Ok(())
    }

    unsafe fn argmax(&self, x: Region) -> Result<Vec<u32>> {
        // SAFETY: the caller upholds the `Backend` contract.
        let data = unsafe { self.view(x)? };
        data.chunks_exact(x.cols.max(1))
            .map(|row| {
                let mut best = 0;
                for (i, &value) in row.iter().enumerate() {
                    if value.is_nan() {
                        return Err(EngineError::NaN { op: "argmax" });
                    }
                    // Strictly greater: the first maximum wins, as in torch.
                    if value > row[best] {
                        best = i;
                    }
                }
                Ok(best as u32)
            })
            .collect()
    }

    unsafe fn read(&self, x: Region) -> Result<Vec<f32>> {
        // SAFETY: the caller upholds the `Backend` contract.
        Ok(unsafe { self.view(x)? }.to_vec())
    }
}

fn attention_inputs<'a>(
    pool: &CpuKvPool,
    query_cols: usize,
    block_size: usize,
    block_tables: &'a [Vec<u32>],
    row_seq: &'a [usize],
    row_num_keys: &'a [usize],
) -> AttentionInputs<'a> {
    let layout = pool.layout;
    AttentionInputs {
        num_heads: query_cols / layout.head_dim,
        num_kv_heads: layout.kv_heads,
        head_dim: layout.head_dim,
        block_size,
        block_tables,
        row_seq,
        row_num_keys,
    }
}

#[cfg(test)]
mod tests;
