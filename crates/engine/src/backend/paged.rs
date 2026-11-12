//! Paged KV addressing and per-step batch metadata (ADR 0004).
//!
//! **Paged KV.** The KV cache is a pool of fixed-size *blocks* (16 tokens
//! each by default). A sequence doesn't own one contiguous buffer; it owns a
//! *block table*, the list of physical block ids holding its tokens in order,
//! like an OS page table. Token `pos` of a sequence lives at *slot*
//! `block_table[pos / block_size] * block_size + pos % block_size`
//! ([`slot_of`]). That one line is all the address arithmetic paging needs.
//!
//! **Who owns what (ADR 0003).** The *engine* decides which blocks a sequence
//! gets and builds the metadata below. The *backend* owns the KV bytes and
//! only follows the addresses it is given.
//!
//! **One step's batch.** Prefill processes many new tokens per sequence;
//! decode processes exactly one. Either way the step's tokens are laid out as
//! *rows*, sequence after sequence with no padding, and each row has:
//! - a **position** (for RoPE and causality),
//! - a **slot** where its K and V are written,
//! - a **sequence** whose block table its attention reads through,
//! - a **number of keys** it attends to (position + 1: itself and everything
//!   before it, including tokens cached by earlier steps).
//!
//! These are plain arrays on purpose: they are what a GPU kernel receives.

/// Shape of the KV pool. Per layer, K and V are each
/// `[num_blocks * block_size, kv_heads, head_dim]`: a flat array of slots,
/// where slot `block * block_size + i` is position `i` within physical `block`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    pub num_layers: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl KvLayout {
    pub fn num_slots(&self) -> usize {
        self.num_blocks * self.block_size
    }
    /// f32 elements in one layer's K (or V).
    pub fn layer_elems(&self) -> usize {
        self.num_slots() * self.kv_heads * self.head_dim
    }
}

/// Physical slot of logical position `pos` in a sequence.
pub fn slot_of(block_table: &[u32], pos: usize, block_size: usize) -> u32 {
    let block = block_table[pos / block_size];
    let offset_in_block = (pos % block_size) as u32;
    block * block_size as u32 + offset_in_block
}

/// A block table that doesn't cover all the tokens it's asked to address.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{op}: block table too short: {tokens} tokens need {needed} blocks, table has {have}")]
pub struct BlockTableTooShort {
    pub op: &'static str,
    pub tokens: usize,
    pub needed: usize,
    pub have: usize,
}

fn check_covers(
    op: &'static str,
    block_table: &[u32],
    tokens: usize,
    block_size: usize,
) -> Result<(), BlockTableTooShort> {
    let needed = tokens.div_ceil(block_size);
    if block_table.len() < needed {
        return Err(BlockTableTooShort {
            op,
            tokens,
            needed,
            have: block_table.len(),
        });
    }
    Ok(())
}

/// One sequence's part of a prefill step.
#[derive(Debug, Clone)]
pub struct PrefillSeq<'a> {
    /// Tokens already in the KV cache from earlier steps. 0 in M0; a prefix
    /// cache hit in M4 (only the uncached tail is prefilled).
    pub context_len: usize,
    /// New tokens processed this step.
    pub num_new: usize,
    /// Physical block ids in logical order. Must cover `context_len + num_new`.
    pub block_table: &'a [u32],
}

/// Everything prefill's ops need, flattened into per-row and per-sequence
/// arrays. S sequences with `num_new` tokens each make T = Σ num_new rows.
#[derive(Debug, Clone)]
pub struct PrefillMeta {
    pub block_size: usize,
    /// `[S][blocks]`
    pub block_tables: Vec<Vec<u32>>,
    /// `[S]`
    pub context_lens: Vec<usize>,
    /// `[S + 1]`: sequence `s` owns rows `seq_row_starts[s]..seq_row_starts[s + 1]`.
    /// (FlashAttention calls this `cu_seqlens`, cumulative sequence lengths.)
    pub seq_row_starts: Vec<usize>,
    /// `[T]`: absolute position of each row (`context_len + i`).
    pub positions: Vec<u32>,
    /// `[T]`: slot each row's K and V are written to.
    pub slot_mapping: Vec<u32>,
    /// `[T]`: which sequence each row belongs to.
    pub row_seq: Vec<usize>,
    /// `[T]`: keys each row attends to (`position + 1`).
    pub row_num_keys: Vec<usize>,
}

impl PrefillMeta {
    pub fn new(seqs: &[PrefillSeq<'_>], block_size: usize) -> Result<Self, BlockTableTooShort> {
        let mut meta = PrefillMeta {
            block_size,
            block_tables: Vec::with_capacity(seqs.len()),
            context_lens: Vec::with_capacity(seqs.len()),
            seq_row_starts: vec![0],
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            row_seq: Vec::new(),
            row_num_keys: Vec::new(),
        };
        for (seq_index, seq) in seqs.iter().enumerate() {
            let total_len = seq.context_len + seq.num_new;
            check_covers("prefill", seq.block_table, total_len, block_size)?;
            for pos in seq.context_len..total_len {
                meta.positions.push(pos as u32);
                meta.slot_mapping
                    .push(slot_of(seq.block_table, pos, block_size));
                meta.row_seq.push(seq_index);
                meta.row_num_keys.push(pos + 1);
            }
            meta.seq_row_starts.push(meta.positions.len());
            meta.context_lens.push(seq.context_len);
            meta.block_tables.push(seq.block_table.to_vec());
        }
        Ok(meta)
    }

    pub fn num_seqs(&self) -> usize {
        self.context_lens.len()
    }

    pub fn num_rows(&self) -> usize {
        self.positions.len()
    }

    /// Each sequence's last row: the one whose output predicts the next token.
    pub fn last_rows(&self) -> Vec<u32> {
        self.seq_row_starts[1..]
            .iter()
            .map(|&next_start| (next_start - 1) as u32)
            .collect()
    }
}

/// One sequence's part of a decode step: exactly one new token.
#[derive(Debug, Clone)]
pub struct DecodeSeq<'a> {
    /// Tokens already in the KV cache. The new token goes at this position.
    pub context_len: usize,
    pub block_table: &'a [u32],
}

/// Everything decode's ops need. Row `s` is sequence `s`'s one new token, so
/// rows and sequences coincide (T = S).
#[derive(Debug, Clone)]
pub struct DecodeMeta {
    pub block_size: usize,
    /// `[S][blocks]`
    pub block_tables: Vec<Vec<u32>>,
    /// `[S]`: position of each new token (= its sequence's `context_len`).
    pub positions: Vec<u32>,
    /// `[S]`: slot each new token's K and V are written to.
    pub slot_mapping: Vec<u32>,
    /// `[S]`: always `0, 1, 2, …`. Kept so attention takes the same inputs
    /// for prefill and decode.
    pub row_seq: Vec<usize>,
    /// `[S]`: keys each row attends to, including the new token (`context_len + 1`).
    pub row_num_keys: Vec<usize>,
}

impl DecodeMeta {
    pub fn new(seqs: &[DecodeSeq<'_>], block_size: usize) -> Result<Self, BlockTableTooShort> {
        let mut meta = DecodeMeta {
            block_size,
            block_tables: Vec::with_capacity(seqs.len()),
            positions: Vec::with_capacity(seqs.len()),
            slot_mapping: Vec::with_capacity(seqs.len()),
            row_seq: Vec::with_capacity(seqs.len()),
            row_num_keys: Vec::with_capacity(seqs.len()),
        };
        for (seq_index, seq) in seqs.iter().enumerate() {
            let pos = seq.context_len;
            check_covers("decode", seq.block_table, pos + 1, block_size)?;
            meta.block_tables.push(seq.block_table.to_vec());
            meta.positions.push(pos as u32);
            meta.slot_mapping
                .push(slot_of(seq.block_table, pos, block_size));
            meta.row_seq.push(seq_index);
            meta.row_num_keys.push(pos + 1);
        }
        Ok(meta)
    }

    pub fn num_seqs(&self) -> usize {
        self.positions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_of_follows_block_table() {
        // Logical blocks 0, 1, 2 live in physical blocks 7, 2, 5. Block size 4.
        let table = [7, 2, 5];
        assert_eq!(slot_of(&table, 0, 4), 28); // block 7, offset 0
        assert_eq!(slot_of(&table, 3, 4), 31); // block 7, offset 3
        assert_eq!(slot_of(&table, 4, 4), 8); //  block 2, offset 0
        assert_eq!(slot_of(&table, 9, 4), 21); // block 5, offset 1
    }

    #[test]
    fn prefill_meta_flattens_a_batch() {
        let table_a = [3, 0];
        let table_b = [1, 4, 6];
        let meta = PrefillMeta::new(
            &[
                // a: new sequence, 5 tokens.
                PrefillSeq {
                    context_len: 0,
                    num_new: 5,
                    block_table: &table_a,
                },
                // b: 6 tokens already cached, 3 new.
                PrefillSeq {
                    context_len: 6,
                    num_new: 3,
                    block_table: &table_b,
                },
            ],
            4,
        )
        .unwrap();
        assert_eq!(meta.seq_row_starts, [0, 5, 8]);
        assert_eq!(meta.positions, [0, 1, 2, 3, 4, 6, 7, 8]);
        // a: positions 0–3 in block 3 (slots 12–15), position 4 in block 0 (slot 0).
        // b: positions 6, 7 in block 4 (slots 18, 19), position 8 in block 6 (slot 24).
        assert_eq!(meta.slot_mapping, [12, 13, 14, 15, 0, 18, 19, 24]);
        assert_eq!(meta.row_seq, [0, 0, 0, 0, 0, 1, 1, 1]);
        // b's rows see its 6 cached keys too.
        assert_eq!(meta.row_num_keys, [1, 2, 3, 4, 5, 7, 8, 9]);
        assert_eq!(meta.last_rows(), [4, 7]);
    }

    #[test]
    fn decode_meta_one_row_per_sequence() {
        let table_a = [9];
        let table_b = [2, 5];
        let meta = DecodeMeta::new(
            &[
                DecodeSeq {
                    context_len: 0,
                    block_table: &table_a,
                },
                DecodeSeq {
                    context_len: 6,
                    block_table: &table_b,
                },
            ],
            4,
        )
        .unwrap();
        assert_eq!(meta.positions, [0, 6]);
        assert_eq!(meta.slot_mapping, [36, 22]); // block 9 offset 0; block 5 offset 2
        assert_eq!(meta.row_seq, [0, 1]);
        assert_eq!(meta.row_num_keys, [1, 7]);
    }

    #[test]
    fn short_block_table_is_rejected() {
        let one_block = [0];
        let err = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 0,
                num_new: 5,
                block_table: &one_block,
            }],
            4,
        )
        .unwrap_err();
        assert_eq!(
            err,
            BlockTableTooShort {
                op: "prefill",
                tokens: 5,
                needed: 2,
                have: 1
            }
        );
        // Decode at position 4 needs a second block too.
        let err = DecodeMeta::new(
            &[DecodeSeq {
                context_len: 4,
                block_table: &one_block,
            }],
            4,
        )
        .unwrap_err();
        assert_eq!(err.needed, 2);
    }
}
