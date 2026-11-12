//! The engine's error type. One enum for the whole crate: every layer (paged
//! metadata, arena, backend ops, model, generator) reports through it, and a
//! caller handles one type. Split by layer only if a caller ever needs to
//! match on "which layer failed".

use crate::backend::BlockTableTooShort;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum EngineError {
    // ---- shape and input checks (arena ops, model) ----
    #[error("{op}: {msg}")]
    Shape { op: &'static str, msg: String },
    #[error("{what}: need {need}, limit {limit}")]
    Capacity {
        what: &'static str,
        need: usize,
        limit: usize,
    },

    // ---- paged KV addressing ----
    #[error(transparent)]
    BlockTable(#[from] BlockTableTooShort),
    #[error("{op}: slot {slot} out of range (pool has {num_slots})")]
    SlotOutOfRange {
        op: &'static str,
        slot: usize,
        num_slots: usize,
    },
    #[error("{op}: block {block} out of range (pool has {num_blocks})")]
    BlockOutOfRange {
        op: &'static str,
        block: usize,
        num_blocks: usize,
    },

    // ---- activation memory ----
    #[error(
        "arena exhausted: need {need} elements at offset {top}, capacity {capacity} \
         (the size arithmetic in model::arena_elems is wrong)"
    )]
    ArenaExhausted {
        need: usize,
        top: usize,
        capacity: usize,
    },
    #[error("tensor belongs to a different arena")]
    WrongArena,
    #[error("region outside the workspace (workspace not allocated, or too small)")]
    OutsideWorkspace,

    // ---- numerics ----
    #[error("rope: position {pos} beyond the precomputed table ({max})")]
    PositionOutOfRange { pos: usize, max: usize },
    #[error("{op}: NaN in input")]
    NaN { op: &'static str },
}

pub type Result<T> = std::result::Result<T, EngineError>;
