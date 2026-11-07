//! Inference engine: model loading, forward pass, KV cache, scheduler.
//!
//! So far: config and weights (M0 PR1); paged KV addressing (M0 PR2).

pub mod backend;
pub mod config;
pub mod weights;

pub use config::{DType, LLAMA_3_2_1B, ModelConfig, RopeScaling};
pub use weights::{LayerWeights, LoadError, ModelWeights, Weight};
