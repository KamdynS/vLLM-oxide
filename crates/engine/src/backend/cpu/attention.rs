//! Paged attention on the CPU: the reference the CUDA kernels (M1) are tested
//! against. Deliberately the textbook two-pass softmax, not FlashAttention's
//! online softmax: it should be obviously correct, not fast.
//!
//! Math, for one query row at position `p` and one query head `h`:
//!
//! ```text
//! kv_head  = h / (heads / kv_heads)                     GQA: query heads share a KV head
//! score_j  = (q_h · K[j, kv_head]) / sqrt(head_dim)     for every key j in 0..=p
//! weight_j = exp(score_j − max) / Σ_k exp(score_k − max)  softmax (max subtracted for stability)
//! out_h    = Σ_j weight_j · V[j, kv_head]
//! ```
//!
//! Key `j` of a sequence is stored at slot `block_table[j / block_size] *
//! block_size + j % block_size` of the KV pool (see `backend::slot_of`).
//! Causality needs no mask: a query at position `p` is only given keys `0..=p`.
//!
//! Bottleneck: memory. Every key and value of the sequence is read for every
//! query; at decode that is the whole sequence's KV cache per new token.

use rayon::prelude::*;

use super::{CpuKvPool, dot};
use crate::backend::{EngineError, Result, slot_of};

/// What one attention call needs to know about each query row. The same plain
/// arrays a GPU kernel will receive: no closures, nothing CPU-specific.
pub(super) struct AttentionInputs<'a> {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    /// `[num_seqs][blocks]`: each sequence's block table.
    pub block_tables: &'a [Vec<u32>],
    /// `[rows]`: which sequence (index into `block_tables`) each row belongs to.
    pub row_seq: &'a [usize],
    /// `[rows]`: how many keys each row attends to (its position + 1).
    pub row_num_keys: &'a [usize],
}

/// Attention for every row of `queries` `[rows, num_heads * head_dim]`,
/// written to `output` (same shape). Rows are independent, so they run in
/// parallel.
pub(super) fn paged_attention(
    queries: &[f32],
    output: &mut [f32],
    pool: &CpuKvPool,
    layer: usize,
    inputs: &AttentionInputs<'_>,
) {
    let row_width = inputs.num_heads * inputs.head_dim;
    // `for_each_init` gives each rayon worker one reusable scores buffer,
    // instead of allocating one per row.
    output
        .par_chunks_mut(row_width)
        .zip(queries.par_chunks(row_width))
        .enumerate()
        .for_each_init(Vec::new, |scores, (row, (out_row, query_row))| {
            let block_table = &inputs.block_tables[inputs.row_seq[row]];
            let num_keys = inputs.row_num_keys[row];
            attend_one_row(
                query_row,
                out_row,
                pool,
                layer,
                inputs,
                block_table,
                num_keys,
                scores,
            );
        });
}

/// Attention for one query row, all heads.
#[allow(clippy::too_many_arguments)]
fn attend_one_row(
    query_row: &[f32],   // [num_heads * head_dim]
    out_row: &mut [f32], // [num_heads * head_dim]
    pool: &CpuKvPool,
    layer: usize,
    inputs: &AttentionInputs<'_>,
    block_table: &[u32],
    num_keys: usize,
    scores: &mut Vec<f32>, // scratch, reused across heads and rows
) {
    let head_dim = inputs.head_dim;
    let heads_per_kv_head = inputs.num_heads / inputs.num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let keys = &pool.keys[layer];
    let values = &pool.values[layer];
    let slot_width = inputs.num_kv_heads * head_dim; // f32s per slot in the pool

    for head in 0..inputs.num_heads {
        let kv_head = head / heads_per_kv_head;
        let query = &query_row[head * head_dim..][..head_dim];
        // Where key/value `j` of this sequence, for this KV head, starts in the pool.
        let pool_offset = |j: usize| {
            let slot = slot_of(block_table, j, inputs.block_size) as usize;
            slot * slot_width + kv_head * head_dim
        };

        // Pass 1: a score per key.
        scores.clear();
        for j in 0..num_keys {
            let key = &keys[pool_offset(j)..][..head_dim];
            scores.push(dot(query, key) * scale);
        }

        // Softmax: subtract the max so exp() can't overflow.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
            total += *score;
        }

        // Pass 2: weighted sum of values.
        let out_head = &mut out_row[head * head_dim..][..head_dim];
        out_head.fill(0.0);
        for (j, &unnormalized) in scores.iter().enumerate() {
            let weight = unnormalized / total;
            let value = &values[pool_offset(j)..][..head_dim];
            for (out, &v) in out_head.iter_mut().zip(value) {
                *out += weight * v;
            }
        }
    }
}

/// Every block id in every table must be inside the pool. Checked once per
/// call, so the per-key indexing above can't go out of bounds.
pub(super) fn check_block_tables(
    op: &'static str,
    block_tables: &[Vec<u32>],
    num_blocks: usize,
) -> Result<()> {
    for table in block_tables {
        if let Some(&block) = table.iter().find(|&&b| b as usize >= num_blocks) {
            return Err(EngineError::BlockOutOfRange {
                op,
                block: block as usize,
                num_blocks,
            });
        }
    }
    Ok(())
}
