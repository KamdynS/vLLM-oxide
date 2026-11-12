//! Llama weights loaded from `model.safetensors` into host f32 buffers.
//!
//! The file stores bf16. We convert to f32 once at load (ADR 0001: the CPU
//! backend computes in f32). bf16 is exactly the top 16 bits of an f32, so the
//! conversion is a shift and loses nothing.
//!
//! Matrices keep HF's `[out_features, in_features]` row-major layout: output
//! `i` of a linear layer is the dot product of the input with row `i`, which is
//! contiguous. That is what a decode-time vector × matrix reads front to back.

use std::collections::HashSet;
use std::path::Path;

use safetensors::{Dtype, SafeTensors};

use crate::config::ModelConfig;

/// One weight tensor in host memory. Not the general tensor type; the
/// forward-pass PR decides that.
#[derive(Debug, Clone, PartialEq)]
pub struct Weight {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Weight {
    /// Build a weight from raw data. Used by tests to make small random models.
    /// Fails if `data.len()` doesn't match the product of `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Option<Self> {
        (shape.iter().product::<usize>() == data.len()).then_some(Weight { shape, data })
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Weights of one decoder layer. Shapes for Llama 3.2 1B in comments.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Weight, // [hidden]                       [2048]
    pub wq: Weight,        // [n_heads * head_dim, hidden]    [2048, 2048]
    pub wk: Weight,        // [n_kv_heads * head_dim, hidden] [512, 2048]
    pub wv: Weight,        // [n_kv_heads * head_dim, hidden] [512, 2048]
    pub wo: Weight,        // [hidden, n_heads * head_dim]    [2048, 2048]
    pub mlp_norm: Weight,  // [hidden]                       [2048]
    pub w_gate: Weight,    // [intermediate, hidden]          [8192, 2048]
    pub w_up: Weight,      // [intermediate, hidden]          [8192, 2048]
    pub w_down: Weight,    // [hidden, intermediate]          [2048, 8192]
}

/// All model weights. `embed` is tied: it maps token ids to vectors at the
/// input *and* is the LM head at the output, so it is stored once.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub embed: Weight, // [vocab, hidden] [128256, 2048]
    pub layers: Vec<LayerWeights>,
    pub final_norm: Weight, // [hidden] [2048]
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("parsing safetensors: {0}")]
    Format(#[from] safetensors::SafeTensorError),
    #[error("missing tensor {0}")]
    Missing(String),
    #[error("tensor {name}: expected shape {expected:?}, file has {actual:?}")]
    Shape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor {name}: expected dtype BF16, file has {actual:?}")]
    Dtype { name: String, actual: Dtype },
    #[error("config says tie_word_embeddings, but the file also has lm_head.weight")]
    UntiedLmHead,
    #[error("unexpected tensors in file: {0:?}")]
    Unexpected(Vec<String>),
}

/// bf16 → f32: place the 16 bits in the top half of an f32. Exact.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Pulls tensors out of the file by name, checking dtype and shape, and
/// remembers which names were taken so leftovers can be reported.
struct Reader<'a> {
    st: SafeTensors<'a>,
    taken: HashSet<String>,
}

impl Reader<'_> {
    fn take(&mut self, name: &str, expected: &[usize]) -> Result<Weight, LoadError> {
        let view = self
            .st
            .tensor(name)
            .map_err(|_| LoadError::Missing(name.to_string()))?;
        if view.dtype() != Dtype::BF16 {
            return Err(LoadError::Dtype {
                name: name.to_string(),
                actual: view.dtype(),
            });
        }
        if view.shape() != expected {
            return Err(LoadError::Shape {
                name: name.to_string(),
                expected: expected.to_vec(),
                actual: view.shape().to_vec(),
            });
        }
        // safetensors stores little-endian; TensorView::new already checked
        // that data.len() == numel * 2, so as_chunks leaves no remainder.
        let (pairs, _) = view.data().as_chunks::<2>();
        let data = pairs
            .iter()
            .map(|&b| bf16_to_f32(u16::from_le_bytes(b)))
            .collect();
        self.taken.insert(name.to_string());
        Ok(Weight {
            shape: expected.to_vec(),
            data,
        })
    }
}

impl ModelWeights {
    /// Load `<model_dir>/model.safetensors`.
    ///
    /// Reads the whole file into memory first (no mmap, so no `unsafe`).
    /// Peak RAM is file size + f32 weights: ~2.5 + 4.9 GB for Llama 3.2 1B.
    pub fn load(model_dir: &Path, cfg: &ModelConfig) -> Result<Self, LoadError> {
        let path = model_dir.join("model.safetensors");
        let bytes = std::fs::read(&path).map_err(|source| LoadError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_safetensors_bytes(&bytes, cfg)
    }

    pub fn from_safetensors_bytes(bytes: &[u8], cfg: &ModelConfig) -> Result<Self, LoadError> {
        let st = SafeTensors::deserialize(bytes)?;
        if st.tensor("lm_head.weight").is_ok() {
            return Err(LoadError::UntiedLmHead);
        }
        let mut r = Reader {
            st,
            taken: HashSet::new(),
        };

        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
        let ffn = cfg.intermediate_size;

        let embed = r.take("model.embed_tokens.weight", &[cfg.vocab_size, h])?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                attn_norm: r.take(&format!("{p}.input_layernorm.weight"), &[h])?,
                wq: r.take(&format!("{p}.self_attn.q_proj.weight"), &[q_dim, h])?,
                wk: r.take(&format!("{p}.self_attn.k_proj.weight"), &[kv_dim, h])?,
                wv: r.take(&format!("{p}.self_attn.v_proj.weight"), &[kv_dim, h])?,
                wo: r.take(&format!("{p}.self_attn.o_proj.weight"), &[h, q_dim])?,
                mlp_norm: r.take(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                w_gate: r.take(&format!("{p}.mlp.gate_proj.weight"), &[ffn, h])?,
                w_up: r.take(&format!("{p}.mlp.up_proj.weight"), &[ffn, h])?,
                w_down: r.take(&format!("{p}.mlp.down_proj.weight"), &[h, ffn])?,
            });
        }
        let final_norm = r.take("model.norm.weight", &[h])?;

        let mut leftover: Vec<String> =
            r.st.names()
                .into_iter()
                .filter(|n| !r.taken.contains(*n))
                .map(str::to_string)
                .collect();
        if !leftover.is_empty() {
            leftover.sort();
            return Err(LoadError::Unexpected(leftover));
        }

        Ok(ModelWeights {
            embed,
            layers,
            final_norm,
        })
    }

    /// Every weight tensor, in file order, with its HF name.
    pub fn named(&self) -> Vec<(String, &Weight)> {
        let mut out = vec![("model.embed_tokens.weight".to_string(), &self.embed)];
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            out.extend([
                (format!("{p}.input_layernorm.weight"), &l.attn_norm),
                (format!("{p}.self_attn.q_proj.weight"), &l.wq),
                (format!("{p}.self_attn.k_proj.weight"), &l.wk),
                (format!("{p}.self_attn.v_proj.weight"), &l.wv),
                (format!("{p}.self_attn.o_proj.weight"), &l.wo),
                (format!("{p}.post_attention_layernorm.weight"), &l.mlp_norm),
                (format!("{p}.mlp.gate_proj.weight"), &l.w_gate),
                (format!("{p}.mlp.up_proj.weight"), &l.w_up),
                (format!("{p}.mlp.down_proj.weight"), &l.w_down),
            ]);
        }
        out.push(("model.norm.weight".to_string(), &self.final_norm));
        out
    }

    /// Parameters actually loaded. Must equal `ModelConfig::param_count`.
    pub fn param_count(&self) -> usize {
        self.named().iter().map(|(_, w)| w.numel()).sum()
    }
}

#[cfg(test)]
mod tests {
    //! Weight loading against small synthetic safetensors files built in-test.

    use super::*;
    use crate::config::LLAMA_3_2_1B;
    use safetensors::tensor::TensorView;

    /// A 2-layer model small enough to build in memory. Sizes are arbitrary but
    /// keep GQA (4 query heads, 2 KV heads, head_dim 3) and non-power-of-two dims.
    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 12,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 20,
            num_hidden_layers: 2,
            vocab_size: 10,
            ..LLAMA_3_2_1B
        }
    }

    /// (name, shape) for every tensor a Llama checkpoint with `c` should contain.
    fn expected_tensors(c: &ModelConfig) -> Vec<(String, Vec<usize>)> {
        let h = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim();
        let kv = c.num_key_value_heads * c.head_dim();
        let f = c.intermediate_size;
        let mut t = vec![(
            "model.embed_tokens.weight".to_string(),
            vec![c.vocab_size, h],
        )];
        for i in 0..c.num_hidden_layers {
            let p = format!("model.layers.{i}");
            t.extend([
                (format!("{p}.input_layernorm.weight"), vec![h]),
                (format!("{p}.self_attn.q_proj.weight"), vec![q, h]),
                (format!("{p}.self_attn.k_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.v_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.o_proj.weight"), vec![h, q]),
                (format!("{p}.post_attention_layernorm.weight"), vec![h]),
                (format!("{p}.mlp.gate_proj.weight"), vec![f, h]),
                (format!("{p}.mlp.up_proj.weight"), vec![f, h]),
                (format!("{p}.mlp.down_proj.weight"), vec![h, f]),
            ]);
        }
        t.push(("model.norm.weight".to_string(), vec![h]));
        t
    }

    /// Deterministic, distinct, finite bf16 bit patterns per tensor, covering
    /// both signs and a spread of exponents.
    fn bf16_bits(tensor_idx: usize, n: usize) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let sign = ((i + tensor_idx) % 2) as u16;
                let exp = 100 + ((i * 7 + tensor_idx * 3) % 50) as u16; // 100..150: finite, normal
                let mantissa = ((i * 13 + tensor_idx) % 128) as u16;
                (sign << 15) | (exp << 7) | mantissa
            })
            .collect()
    }

    /// A tensor to write: (name, dtype, shape, raw little-endian bytes).
    type Part = (String, Dtype, Vec<usize>, Vec<u8>);

    /// Serialize tensors (name, dtype, shape, raw bytes) into a safetensors buffer.
    fn build(tensors: &[Part]) -> Vec<u8> {
        let views = tensors
            .iter()
            .map(|(n, d, s, b)| (n.clone(), TensorView::new(*d, s.clone(), b).unwrap()));
        safetensors::serialize(views, None).unwrap()
    }

    /// The full valid checkpoint for `c`, as editable parts. Also returns the bits.
    fn valid_parts(c: &ModelConfig) -> (Vec<Part>, Vec<Vec<u16>>) {
        let mut parts = Vec::new();
        let mut all_bits = Vec::new();
        for (idx, (name, shape)) in expected_tensors(c).into_iter().enumerate() {
            let bits = bf16_bits(idx, shape.iter().product());
            let bytes = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            parts.push((name, Dtype::BF16, shape, bytes));
            all_bits.push(bits);
        }
        (parts, all_bits)
    }

    #[test]
    fn loads_valid_checkpoint_with_exact_values() {
        let c = tiny_config();
        let (parts, bits) = valid_parts(&c);
        let w = ModelWeights::from_safetensors_bytes(&build(&parts), &c).unwrap();

        assert_eq!(w.layers.len(), 2);
        assert_eq!(w.param_count(), c.param_count());

        let named = w.named();
        assert_eq!(named.len(), parts.len());
        for ((name, weight), ((exp_name, _, exp_shape, _), exp_bits)) in
            named.iter().zip(parts.iter().zip(&bits))
        {
            assert_eq!(name, exp_name);
            assert_eq!(weight.shape(), exp_shape.as_slice(), "{name}");
            for (i, (&got, &want)) in weight.data().iter().zip(exp_bits).enumerate() {
                // bf16 → f32 is exact: the low 16 bits are zero and the top 16 are
                // the original bf16.
                assert_eq!(got.to_bits() >> 16, want as u32, "{name}[{i}]");
                assert_eq!(got.to_bits() & 0xFFFF, 0, "{name}[{i}]");
            }
        }
    }

    #[test]
    fn bf16_conversion_known_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x3E80), 0.25);
    }

    #[test]
    fn missing_tensor_is_named() {
        let c = tiny_config();
        let (mut parts, _) = valid_parts(&c);
        parts.retain(|p| p.0 != "model.layers.1.mlp.up_proj.weight");
        match ModelWeights::from_safetensors_bytes(&build(&parts), &c) {
            Err(LoadError::Missing(n)) => assert_eq!(n, "model.layers.1.mlp.up_proj.weight"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn wrong_shape_is_reported() {
        let c = tiny_config();
        let (mut parts, _) = valid_parts(&c);
        // Transposed k_proj: same element count, wrong layout.
        let k = parts
            .iter_mut()
            .find(|p| p.0 == "model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        k.2 = vec![12, 6];
        match ModelWeights::from_safetensors_bytes(&build(&parts), &c) {
            Err(LoadError::Shape {
                name,
                expected,
                actual,
            }) => {
                assert_eq!(name, "model.layers.0.self_attn.k_proj.weight");
                assert_eq!(expected, vec![6, 12]);
                assert_eq!(actual, vec![12, 6]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn non_bf16_dtype_is_rejected() {
        let c = tiny_config();
        let (mut parts, _) = valid_parts(&c);
        let n = &mut parts[1]; // layer 0 input_layernorm, shape [12]
        n.1 = Dtype::F32;
        n.3 = vec![0u8; 12 * 4];
        assert!(matches!(
            ModelWeights::from_safetensors_bytes(&build(&parts), &c),
            Err(LoadError::Dtype { .. })
        ));
    }

    #[test]
    fn lm_head_contradicts_tied_config() {
        let c = tiny_config();
        let (mut parts, _) = valid_parts(&c);
        let embed = parts[0].clone();
        parts.push(("lm_head.weight".to_string(), embed.1, embed.2, embed.3));
        assert!(matches!(
            ModelWeights::from_safetensors_bytes(&build(&parts), &c),
            Err(LoadError::UntiedLmHead)
        ));
    }

    #[test]
    fn extra_tensors_are_rejected() {
        let c = tiny_config();
        let (mut parts, _) = valid_parts(&c);
        // A third layer the config doesn't know about.
        let extra = parts[1].clone();
        parts.push((
            "model.layers.2.input_layernorm.weight".to_string(),
            extra.1,
            extra.2,
            extra.3,
        ));
        match ModelWeights::from_safetensors_bytes(&build(&parts), &c) {
            Err(LoadError::Unexpected(names)) => {
                assert_eq!(names, vec!["model.layers.2.input_layernorm.weight"])
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn garbage_bytes_fail_cleanly() {
        let c = tiny_config();
        assert!(matches!(
            ModelWeights::from_safetensors_bytes(b"not a safetensors file", &c),
            Err(LoadError::Format(_))
        ));
    }
}
