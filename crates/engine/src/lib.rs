//! Inference engine: model loading, forward pass, KV cache, scheduler.
//!
//! So far: config and weights (M0 PR1); paged KV addressing, the backend
//! trait and the engine error type (M0 PR2).

pub mod backend;
pub mod config;
pub mod error;
pub mod weights;

pub use backend::Backend;
pub use config::{DType, LLAMA_3_2_1B, ModelConfig, RopeScaling};
pub use error::{EngineError, Result};
pub use weights::{LayerWeights, LoadError, ModelWeights, Weight};
