//! Inference engine: model loading, forward pass, KV cache, scheduler.
//!
//! So far: the model config and weight loading (M0 PR1).

pub mod config;
pub mod weights;

pub use config::{DType, LLAMA_3_2_1B, ModelConfig, RopeScaling};
pub use weights::{LayerWeights, LoadError, ModelWeights, Weight};
