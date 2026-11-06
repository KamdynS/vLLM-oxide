//! The model's hyperparameters, written into the code.
//!
//! This engine serves exactly one model, Llama 3.2 1B (ADR 0002), so its config
//! is a `const`, not parsed from `config.json` at runtime. Consistency is checked
//! at compile time (bottom of this file), and `tests/real_model.rs` checks every
//! number against the downloaded `config.json` once.

/// Llama hyperparameters. Field names match HF `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

/// llama3 RoPE frequency scaling (`rope_scaling` in config.json).
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

/// meta-llama/Llama-3.2-1B `config.json`. Also implied by that file and not
/// stored: SiLU MLP, RMSNorm, no attention/MLP bias, tied embeddings.
pub const LLAMA_3_2_1B: ModelConfig = ModelConfig {
    hidden_size: 2048,
    intermediate_size: 8192,
    num_hidden_layers: 16,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    vocab_size: 128256,
    max_position_embeddings: 131072,
    rms_norm_eps: 1e-5,
    rope_theta: 500000.0,
    rope_scaling: RopeScaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192,
    },
    bos_token_id: 128000,
    eos_token_id: 128001,
};

/// Element type of a buffer, for sizing memory (weights, KV cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
        }
    }
}

impl ModelConfig {
    /// Width of one attention head. Llama 3.2 1B: 2048 / 32 = 64.
    pub const fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Query heads that share one KV head (GQA). Llama 3.2 1B: 32 / 8 = 4.
    pub const fn gqa_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Total parameters, counting the tied embedding once.
    /// Llama 3.2 1B: 1,235,814,400.
    pub const fn param_count(&self) -> usize {
        let h = self.hidden_size;
        let q_dim = self.num_attention_heads * self.head_dim();
        let kv_dim = self.num_key_value_heads * self.head_dim();
        let attn = h * q_dim + 2 * h * kv_dim + q_dim * h; // q, k, v, o
        let mlp = 3 * h * self.intermediate_size; // gate, up, down
        let norms = 2 * h; // input_layernorm, post_attention_layernorm
        let embed = self.vocab_size * h;
        let final_norm = h;
        embed + self.num_hidden_layers * (attn + mlp + norms) + final_norm
    }

    /// KV cache bytes one token occupies across all layers: K and V, each
    /// `[num_kv_heads, head_dim]` per layer. Llama 3.2 1B: 64 KiB in f32.
    pub const fn kv_bytes_per_token(&self, dtype: DType) -> usize {
        2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim() * dtype.size_bytes()
    }
}

// Compile-time checks: an inconsistent edit to the const fails the build.
const _: () = {
    let c = &LLAMA_3_2_1B;
    assert!(c.hidden_size.is_multiple_of(c.num_attention_heads));
    assert!(c.num_attention_heads.is_multiple_of(c.num_key_value_heads));
    // The owner's hand calculation (docs/grill-log.md, 2026-11-05).
    assert!(c.head_dim() == 64);
    assert!(c.gqa_group() == 4);
    assert!(c.param_count() == 1_235_814_400);
    assert!(c.kv_bytes_per_token(DType::F32) == 64 * 1024);
    assert!(c.kv_bytes_per_token(DType::BF16) == 32 * 1024);
};
