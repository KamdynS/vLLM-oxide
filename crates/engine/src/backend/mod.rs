//! Device backends. So far: paged KV addressing (`paged`). The backend trait
//! arrives in the next commit.

pub mod paged;

pub use paged::{
    BlockTableTooShort, DecodeMeta, DecodeSeq, KvLayout, PrefillMeta, PrefillSeq, slot_of,
};
