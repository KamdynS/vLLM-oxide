//! Inference engine: model loading, forward pass, KV cache, scheduler.
//!
//! So far: config and weights (M0 PR1); the backend trait, CPU backend,
//! activation arena, Llama forward pass, tokenizer and a single-sequence greedy
//! driver (M0 PR2).

pub mod arena;
pub mod backend;
pub mod config;
pub mod error;
pub mod generate;
pub mod model;
pub mod tokenizer;
pub mod weights;

#[cfg(test)]
pub(crate) mod testutil;

pub use backend::Backend;
pub use backend::cpu::CpuBackend;
pub use config::{DType, LLAMA_3_2_1B, ModelConfig, RopeScaling};
pub use error::{EngineError, Result};
pub use generate::{GenerateOutput, Generator};
pub use model::{EngineConfig, Llama, StepOutput};
pub use tokenizer::Tokenizer;
pub use weights::{LayerWeights, LoadError, ModelWeights, Weight};
