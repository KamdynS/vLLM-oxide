//! Rotary position embedding (RoPE) with Llama 3 frequency scaling.
//!
//! Math: split each head vector of width `d` into halves `x1 = x[..d/2]`,
//! `x2 = x[d/2..]` (HF's "rotate half" pairing, matching how `q_proj` and
//! `k_proj` weights are laid out in HF checkpoints). For pair `i` at position `p`:
//!
//! ```text
//! θ = p · inv_freq[i]
//! x1'[i] = x1[i]·cos θ − x2[i]·sin θ
//! x2'[i] = x2[i]·cos θ + x1[i]·sin θ
//! ```
//!
//! `inv_freq[i] = rope_theta^(−2i/d)`, then Llama 3 scaling: low frequencies
//! (wavelength > original_ctx / low_freq_factor) are divided by `factor`, high
//! frequencies (wavelength < original_ctx / high_freq_factor) are kept, and the
//! band between is interpolated. This is what lets a model trained at 8k context
//! run at 128k.
//!
//! Everything is computed in f32 in the same order as HF
//! (`_compute_llama3_parameters`), so the table matches the reference to the ulp
//! as far as `f32::powf`/`cos` match torch's.

use crate::config::ModelConfig;

/// Precomputed `cos θ` and `sin θ` for every position up to `max_len` and every
/// frequency. `[max_len, head_dim / 2]` each.
#[derive(Debug, Clone)]
pub struct RopeTable {
    pub half: usize,
    pub max_len: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

/// Llama 3 scaled inverse frequencies, `[head_dim / 2]`.
pub fn llama3_inv_freq(cfg: &ModelConfig) -> Vec<f32> {
    let d = cfg.head_dim();
    let base = cfg.rope_theta as f32;
    let rs = &cfg.rope_scaling;
    let factor = rs.factor as f32;
    let low = rs.low_freq_factor as f32;
    let high = rs.high_freq_factor as f32;
    let old_ctx = rs.original_max_position_embeddings as f32;
    let low_freq_wavelen = old_ctx / low;
    let high_freq_wavelen = old_ctx / high;

    (0..d / 2)
        .map(|i| {
            let inv = 1.0 / base.powf((2 * i) as f32 / d as f32);
            let wavelen = 2.0 * std::f32::consts::PI / inv;
            if wavelen < high_freq_wavelen {
                inv // high frequency: untouched
            } else if wavelen > low_freq_wavelen {
                inv / factor // low frequency: fully scaled
            } else {
                let smooth = (old_ctx / wavelen - low) / (high - low);
                (1.0 - smooth) * inv / factor + smooth * inv
            }
        })
        .collect()
}

impl RopeTable {
    pub fn new(cfg: &ModelConfig, max_len: usize) -> Self {
        let inv = llama3_inv_freq(cfg);
        let half = inv.len();
        let mut cos = Vec::with_capacity(max_len * half);
        let mut sin = Vec::with_capacity(max_len * half);
        for p in 0..max_len {
            for &f in &inv {
                // HF: freqs = position (as f32) * inv_freq, in f32.
                let theta = p as f32 * f;
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
        RopeTable {
            half,
            max_len,
            cos,
            sin,
        }
    }

    /// Rotate one row `[heads * head_dim]` in place for position `pos`.
    /// Caller checks `pos < max_len` and `row.len() % (2 * half) == 0`.
    pub fn apply(&self, row: &mut [f32], pos: usize) {
        let h = self.half;
        let cos = &self.cos[pos * h..][..h];
        let sin = &self.sin[pos * h..][..h];
        for head in row.chunks_exact_mut(2 * h) {
            let (x1, x2) = head.split_at_mut(h);
            for i in 0..h {
                let (a, b) = (x1[i], x2[i]);
                x1[i] = a * cos[i] - b * sin[i];
                x2[i] = b * cos[i] + a * sin[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LLAMA_3_2_1B;

    #[test]
    fn llama3_scaling_bands() {
        let inv = llama3_inv_freq(&LLAMA_3_2_1B);
        assert_eq!(inv.len(), 32);
        // i = 0: wavelength 2π, far below 2048 → unscaled, exactly 1.
        assert_eq!(inv[0], 1.0);
        // Lowest frequency: θ^(-62/64), wavelength ≫ 8192 → divided by 32.
        let raw = 1.0 / 500000f32.powf(62.0 / 64.0);
        assert!((inv[31] - raw / 32.0).abs() <= raw * 1e-6);
        // Monotonically decreasing, and every scaled value lies between
        // raw/factor and raw.
        for i in 0..32 {
            let raw = 1.0 / 500000f32.powf((2 * i) as f32 / 64.0);
            assert!(
                inv[i] <= raw && inv[i] >= raw / 32.0 * (1.0 - 1e-6),
                "i={i}"
            );
            if i > 0 {
                assert!(inv[i] < inv[i - 1]);
            }
        }
    }

    #[test]
    fn rotation_matches_direct_formula_and_preserves_norm() {
        let t = RopeTable::new(&LLAMA_3_2_1B, 40);
        let inv = llama3_inv_freq(&LLAMA_3_2_1B);
        let row0: Vec<f32> = (0..128)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) / 3.0)
            .collect();
        let mut row = row0.clone();
        t.apply(&mut row, 37);
        for head in 0..2 {
            for i in 0..32 {
                let th = 37.0f32 * inv[i];
                let (a, b) = (row0[head * 64 + i], row0[head * 64 + 32 + i]);
                let e1 = a * th.cos() - b * th.sin();
                let e2 = b * th.cos() + a * th.sin();
                assert!((row[head * 64 + i] - e1).abs() < 1e-6);
                assert!((row[head * 64 + 32 + i] - e2).abs() < 1e-6);
            }
        }
        let n0: f32 = row0.iter().map(|x| x * x).sum();
        let n1: f32 = row.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-3 * n0);
    }

    #[test]
    fn position_zero_is_identity() {
        let t = RopeTable::new(&LLAMA_3_2_1B, 1);
        let row0: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mut row = row0.clone();
        t.apply(&mut row, 0);
        assert_eq!(row, row0);
    }
}
