//! Helpers shared by unit tests: a tiny Llama config and deterministic random
//! numbers (no `rand` dependency needed for this).

use crate::config::{ModelConfig, RopeScaling};
use crate::weights::{LayerWeights, ModelWeights, Weight};

/// A Llama small enough to run in microseconds, with the same structure as the
/// real one: GQA (6 query heads sharing 2 KV heads, group 3), head_dim 8, a
/// vocab that is not a power of two, 2 layers.
pub fn tiny_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 48,
        intermediate_size: 80,
        num_hidden_layers: 2,
        num_attention_heads: 6,
        num_key_value_heads: 2,
        vocab_size: 37,
        max_position_embeddings: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: RopeScaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 64,
        },
        bos_token_id: 1,
        eos_token_id: 2,
    }
}

/// Deterministic uniform values in [-1, 1) from a 64-bit LCG (Knuth's MMIX
/// constants). Good enough for test data; not for anything statistical.
pub fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

pub fn rand_weight(seed: u64, shape: &[usize], scale: f32) -> Weight {
    let n = shape.iter().product();
    Weight::new(
        shape.to_vec(),
        rand_vec(seed, n).into_iter().map(|x| x * scale).collect(),
    )
    .expect("shape matches data")
}

/// Random weights for `config`. Matrices scaled by 1/sqrt(in) so activations stay
/// O(1) through the layers; norm weights near 1, like a trained model.
pub fn random_weights(config: &ModelConfig, seed: u64) -> ModelWeights {
    let h = config.hidden_size;
    let qd = config.num_attention_heads * config.head_dim();
    let kvd = config.num_key_value_heads * config.head_dim();
    let ffn = config.intermediate_size;
    let m =
        |s: u64, o: usize, i: usize| rand_weight(seed * 1000 + s, &[o, i], 1.0 / (i as f32).sqrt());
    let norm = |s: u64| {
        let d = rand_vec(seed * 1000 + s, h)
            .into_iter()
            .map(|x| 1.0 + 0.1 * x)
            .collect();
        Weight::new(vec![h], d).expect("shape matches data")
    };
    ModelWeights {
        embed: rand_weight(seed * 1000, &[config.vocab_size, h], 1.0),
        layers: (0..config.num_hidden_layers as u64)
            .map(|l| {
                let b = 10 + l * 10;
                LayerWeights {
                    attn_norm: norm(b),
                    wq: m(b + 1, qd, h),
                    wk: m(b + 2, kvd, h),
                    wv: m(b + 3, kvd, h),
                    wo: m(b + 4, h, qd),
                    mlp_norm: norm(b + 5),
                    w_gate: m(b + 6, ffn, h),
                    w_up: m(b + 7, ffn, h),
                    w_down: m(b + 8, h, ffn),
                }
            })
            .collect(),
        final_norm: norm(999),
    }
}

/// Max |a − b| relative to max |b|.
pub fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}
