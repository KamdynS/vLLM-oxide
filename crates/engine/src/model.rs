//! The Llama forward pass, written once against [`Backend`] (ADR 0003).
//!
//! Two entry points, [`Llama::prefill`] and [`Llama::decode`] (ADR 0004).
//! They share every op except attention and how the final rows are picked:
//!
//! ```text
//! residual = embed(tokens)                                     [T, hidden]
//! for each layer:
//!     # attention block
//!     normed  = rmsnorm(residual)
//!     q, k, v = normed·Wqᵀ, normed·Wkᵀ, normed·Wvᵀ              [T, q_dim], [T, kv_dim] ×2
//!     rope(q, k); write k, v into the KV pool at this step's slots
//!     attn    = prefill_attention(q) | decode_attention(q)     [T, q_dim]
//!     residual += attn·Woᵀ
//!     # MLP block (SwiGLU)
//!     normed  = rmsnorm(residual)
//!     gate    = silu(normed·Wgateᵀ) * (normed·Wupᵀ)             [T, intermediate]
//!     residual += gate·Wdownᵀ
//! last   = the rows that predict a next token                  [S, hidden]
//! logits = rmsnorm(last) · embedᵀ      (tied embedding)        [S, vocab]
//! next   = argmax(logits)              (the one sync)          [S]
//! ```
//!
//! T = tokens this step, S = sequences this step. In decode T = S.
//!
//! Memory: `residual` lives in the step's outer arena scope for the whole
//! step. Each attention block and each MLP block gets its own nested scope,
//! freed before the next one starts. [`arena_elems`] computes the peak.

use crate::arena::{Arena, OpStat};
use crate::backend::{Backend, DecodeMeta, EngineError, KvLayout, PrefillMeta, Result};
use crate::config::{DType, ModelConfig};
use crate::weights::ModelWeights;

/// Startup limits, fixed for the engine's lifetime. Each value is a placeholder
/// with a TODO saying when it gets measured.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum tokens processed in one step (new tokens summed across sequences).
    /// Sizes the activation arena. 2048 covers M0's longest prompt (238) with
    /// room for M2's batched prefill. TODO(M2): measure.
    pub max_batch_tokens: usize,
    /// Maximum sequences in one step. 32 is the M2 gate's concurrency target.
    pub max_seqs: usize,
    /// Longest sequence (prompt + output). Sizes the RoPE table. 8192 is far
    /// above M0's ~500 (238 + 256) and is Llama 3's pre-scaling context.
    pub max_model_len: usize,
    /// Tokens per KV block. 16 is vLLM's default; a sequence wastes less than
    /// one block. TODO(M1): measure the attention-kernel tradeoff.
    pub block_size: usize,
    /// KV pool budget. 2 GiB = 32,768 tokens at 64 KiB/token (f32). The 30 GB
    /// box minus 4.9 GB of weights leaves plenty, and it's 64× M0's need.
    pub kv_pool_bytes: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            max_batch_tokens: 2048,
            max_seqs: 32,
            max_model_len: 8192,
            block_size: 16,
            kv_pool_bytes: 2 << 30,
        }
    }
}

/// Peak arena elements (f32) in one step, read off the scope structure in
/// [`Llama::forward`]. T = `max_tokens`, S = `max_seqs`, h = hidden size:
///
/// ```text
/// step scope:       residual                         T·h
/// attention scope:  normed, q, k, v, attn, attn_out  T·(h + q + 2·kv + q + h)
/// MLP scope:        normed, gate, up, down           T·(h + 2·intermediate + h)
/// final:            last, normed, logits             S·(h + h + vocab)
/// peak = T·h + max(attention, MLP, final)
/// ```
///
/// Llama 3.2 1B, T = 2048, S = 32: the MLP scope dominates; 46.1 M f32 =
/// 184 MB. A test checks this equals the arena's measured high-water mark.
pub fn arena_elems(config: &ModelConfig, max_tokens: usize, max_seqs: usize) -> usize {
    let hidden = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim();
    let kv_dim = config.num_key_value_heads * config.head_dim();
    let residual = max_tokens * hidden;
    let attention = max_tokens * (2 * hidden + 2 * q_dim + 2 * kv_dim);
    let mlp = max_tokens * (2 * hidden + 2 * config.intermediate_size);
    let last_rows = max_seqs * (2 * hidden + config.vocab_size);
    residual + attention.max(mlp).max(last_rows)
}

struct LayerWeights<B: Backend> {
    attn_norm: B::Weight,
    q_proj: B::Weight,
    k_proj: B::Weight,
    v_proj: B::Weight,
    o_proj: B::Weight,
    mlp_norm: B::Weight,
    gate_proj: B::Weight,
    up_proj: B::Weight,
    down_proj: B::Weight,
}

struct Weights<B: Backend> {
    /// `[vocab, hidden]`. Also the LM head (tied embedding).
    embed: B::Weight,
    layers: Vec<LayerWeights<B>>,
    final_norm: B::Weight,
}

/// Next token per sequence, and the logits if asked for (tests only).
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// `[S]`
    pub tokens: Vec<u32>,
    /// `[S, vocab]`, row-major. `None` unless requested.
    pub logits: Option<Vec<f32>>,
}

/// Which kind of step is running. Only attention and the final row
/// selection look at this; the rest of the layer is identical.
#[derive(Clone, Copy)]
enum StepKind<'meta> {
    Prefill(&'meta PrefillMeta),
    Decode(&'meta DecodeMeta),
}

pub struct Llama<B: Backend> {
    config: ModelConfig,
    engine_config: EngineConfig,
    backend: B,
    weights: Weights<B>,
    arena: Arena,
}

impl<B: Backend> Llama<B> {
    /// Upload the weights and allocate the activation workspace. The KV pool
    /// is separate ([`Llama::alloc_kv_pool`]) because the engine decides how
    /// it's used.
    pub fn new(
        mut backend: B,
        config: ModelConfig,
        engine_config: EngineConfig,
        weights: ModelWeights,
    ) -> Result<Self> {
        let workspace_elems = arena_elems(
            &config,
            engine_config.max_batch_tokens,
            engine_config.max_seqs,
        );
        backend.alloc_workspace(workspace_elems)?;
        let arena = Arena::new(&backend);

        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in weights.layers {
            layers.push(LayerWeights {
                attn_norm: backend.upload(layer.attn_norm)?,
                q_proj: backend.upload(layer.wq)?,
                k_proj: backend.upload(layer.wk)?,
                v_proj: backend.upload(layer.wv)?,
                o_proj: backend.upload(layer.wo)?,
                mlp_norm: backend.upload(layer.mlp_norm)?,
                gate_proj: backend.upload(layer.w_gate)?,
                up_proj: backend.upload(layer.w_up)?,
                down_proj: backend.upload(layer.w_down)?,
            });
        }
        let weights = Weights {
            embed: backend.upload(weights.embed)?,
            layers,
            final_norm: backend.upload(weights.final_norm)?,
        };
        Ok(Llama {
            config,
            engine_config,
            backend,
            weights,
            arena,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn engine_config(&self) -> &EngineConfig {
        &self.engine_config
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn arena(&self) -> &Arena {
        &self.arena
    }
    pub fn enable_profiling(&mut self) {
        self.arena.enable_profiling();
    }
    pub fn profile(&self) -> Option<std::collections::BTreeMap<&'static str, OpStat>> {
        self.arena.profile()
    }

    /// The KV pool shape for this model and engine config.
    pub fn kv_layout(&self) -> KvLayout {
        let block_bytes =
            self.engine_config.block_size * self.config.kv_bytes_per_token(DType::F32);
        KvLayout {
            num_layers: self.config.num_hidden_layers,
            num_blocks: self.engine_config.kv_pool_bytes / block_bytes,
            block_size: self.engine_config.block_size,
            kv_heads: self.config.num_key_value_heads,
            head_dim: self.config.head_dim(),
        }
    }

    /// Allocate the KV pool. Call once at startup.
    pub fn alloc_kv_pool(&self) -> Result<B::KvPool> {
        self.backend.alloc_kv_pool(self.kv_layout())
    }

    /// Process each sequence's new tokens and return each sequence's next
    /// token. `tokens` `[T]` are laid out as `meta` describes.
    pub fn prefill(
        &mut self,
        kv_pool: &mut B::KvPool,
        tokens: &[u32],
        meta: &PrefillMeta,
        want_logits: bool,
    ) -> Result<StepOutput> {
        self.check_batch(tokens.len(), meta.num_rows(), meta.num_seqs())?;
        self.forward(kv_pool, tokens, StepKind::Prefill(meta), want_logits)
    }

    /// One new token per sequence; return each sequence's next token.
    /// `tokens` `[S]`.
    pub fn decode(
        &mut self,
        kv_pool: &mut B::KvPool,
        tokens: &[u32],
        meta: &DecodeMeta,
        want_logits: bool,
    ) -> Result<StepOutput> {
        self.check_batch(tokens.len(), meta.num_seqs(), meta.num_seqs())?;
        self.forward(kv_pool, tokens, StepKind::Decode(meta), want_logits)
    }

    fn check_batch(&self, tokens: usize, meta_rows: usize, sequences: usize) -> Result<()> {
        if tokens != meta_rows {
            return Err(EngineError::Shape {
                op: "forward",
                msg: format!("{tokens} tokens, metadata describes {meta_rows}"),
            });
        }
        if tokens > self.engine_config.max_batch_tokens {
            return Err(EngineError::Capacity {
                what: "tokens per step",
                need: tokens,
                limit: self.engine_config.max_batch_tokens,
            });
        }
        if sequences > self.engine_config.max_seqs {
            return Err(EngineError::Capacity {
                what: "sequences per step",
                need: sequences,
                limit: self.engine_config.max_seqs,
            });
        }
        Ok(())
    }

    fn forward(
        &mut self,
        kv_pool: &mut B::KvPool,
        tokens: &[u32],
        kind: StepKind<'_>,
        want_logits: bool,
    ) -> Result<StepOutput> {
        // Borrow the fields separately: the arena is borrowed mutably while
        // the backend and weights are read.
        let Llama {
            config,
            backend,
            weights,
            arena,
            ..
        } = self;
        let head_dim = config.head_dim();
        let eps = config.rms_norm_eps as f32;
        // Both modes carry one position and one KV slot per token in this step,
        // with the same meaning; only the attention op (below) differs.
        let (positions, slots) = match kind {
            StepKind::Prefill(meta) => (&meta.positions[..], &meta.slot_mapping[..]),
            StepKind::Decode(meta) => (&meta.positions[..], &meta.slot_mapping[..]),
        };

        arena.scope(&*backend, |step| {
            // [T, hidden]: lives for the whole step. Each block adds into it.
            let mut residual = step.embed(&weights.embed, tokens)?;

            for (layer_index, layer) in weights.layers.iter().enumerate() {
                let _span = tracing::trace_span!("layer", layer_index).entered();

                // Attention block. Its temporaries are freed when the scope closes.
                step.scope(|block| -> Result<()> {
                    let normed = block.rmsnorm(&residual, &layer.attn_norm, eps)?;
                    let mut q = block.matmul(&normed, &layer.q_proj)?; // [T, heads * head_dim]
                    let mut k = block.matmul(&normed, &layer.k_proj)?; // [T, kv_heads * head_dim]
                    let v = block.matmul(&normed, &layer.v_proj)?; //     [T, kv_heads * head_dim]
                    block.rope(&mut q, positions, head_dim)?;
                    block.rope(&mut k, positions, head_dim)?;
                    // Write this step's K and V before attention, so attention
                    // reads every key (cached and new) from the pool the same way.
                    block.write_kv(kv_pool, layer_index, &k, &v, slots)?;
                    let attn = match kind {
                        StepKind::Prefill(meta) => {
                            block.prefill_attention(&q, kv_pool, layer_index, meta)?
                        }
                        StepKind::Decode(meta) => {
                            block.decode_attention(&q, kv_pool, layer_index, meta)?
                        }
                    };
                    let attn_out = block.matmul(&attn, &layer.o_proj)?; // [T, hidden]
                    block.add(&mut residual, &attn_out)
                })?;

                // MLP block (SwiGLU).
                step.scope(|block| -> Result<()> {
                    let normed = block.rmsnorm(&residual, &layer.mlp_norm, eps)?;
                    let mut gate = block.matmul(&normed, &layer.gate_proj)?; // [T, intermediate]
                    let up = block.matmul(&normed, &layer.up_proj)?; //         [T, intermediate]
                    block.silu_mul(&mut gate, &up)?; // gate = silu(gate) * up
                    let down = block.matmul(&gate, &layer.down_proj)?; //       [T, hidden]
                    block.add(&mut residual, &down)
                })?;
            }

            // The LM head runs only on rows that predict a next token (grill
            // Q10): each sequence's last new token. In decode, that's every row.
            let normed = match kind {
                StepKind::Prefill(meta) => {
                    let last = step.gather_rows(&residual, &meta.last_rows())?; // [S, hidden]
                    step.rmsnorm(&last, &weights.final_norm, eps)?
                }
                StepKind::Decode(_) => step.rmsnorm(&residual, &weights.final_norm, eps)?,
            };
            let logits = step.matmul(&normed, &weights.embed)?; // [S, vocab]
            let next_tokens = step.argmax(&logits)?;
            let logits = if want_logits {
                Some(step.read(&logits)?)
            } else {
                None
            };
            Ok(StepOutput {
                tokens: next_tokens,
                logits,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::{DecodeSeq, PrefillSeq};
    use crate::testutil::{max_rel_diff, random_weights, tiny_config};

    fn tiny(engine_config: EngineConfig) -> (Llama<CpuBackend>, crate::backend::cpu::CpuKvPool) {
        let config = tiny_config();
        let b = CpuBackend::new(&config, engine_config.max_model_len);
        let m = Llama::new(b, config.clone(), engine_config, random_weights(&config, 7)).unwrap();
        let kv = m.alloc_kv_pool().unwrap();
        (m, kv)
    }

    fn small_engine_config() -> EngineConfig {
        EngineConfig {
            max_batch_tokens: 64,
            max_seqs: 4,
            max_model_len: 128,
            block_size: 4,
            // tiny KV/token = 2 * 2 layers * 2 heads * 8 * 4 B = 256 B → 32 blocks.
            kv_pool_bytes: 32 * 4 * 256,
        }
    }

    fn prompt(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32)
            .map(|i| (i * 7 + seed * 13 + 3) % 37)
            .collect()
    }

    #[test]
    fn llama_1b_arena_is_184_mb() {
        let e = arena_elems(&crate::config::LLAMA_3_2_1B, 2048, 32);
        assert_eq!(e, 2048 * 2048 + 2048 * (2 * 2048 + 2 * 8192));
        assert_eq!(e * 4 / 1_000_000, 184);
    }

    /// The sizing formula and the real scope structure agree exactly: a full
    /// prefill hits the computed peak, never more.
    #[test]
    fn arena_high_water_equals_formula() {
        let engine_config = small_engine_config();
        let (mut m, mut kv) = tiny(engine_config.clone());
        let p = prompt(64, 1);
        let table: Vec<u32> = (0..16).collect();
        let meta = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 0,
                num_new: 64,
                block_table: &table,
            }],
            4,
        )
        .unwrap();
        m.prefill(&mut kv, &p, &meta, false).unwrap();
        let want = arena_elems(m.config(), 64, 4);
        assert_eq!(m.arena().high_water(), want);
        assert_eq!(m.arena().capacity(), want);
    }

    #[test]
    fn over_limit_batch_is_rejected() {
        let (mut m, mut kv) = tiny(small_engine_config());
        let table: Vec<u32> = (0..17).collect();
        let meta = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 0,
                num_new: 65,
                block_table: &table,
            }],
            4,
        )
        .unwrap();
        let e = m
            .prefill(&mut kv, &prompt(65, 0), &meta, false)
            .unwrap_err();
        assert!(matches!(
            e,
            EngineError::Capacity {
                need: 65,
                limit: 64,
                ..
            }
        ));
    }

    /// Prefill N tokens in one go vs prefill N−k then decode k one at a time:
    /// same next token, logits equal to float noise. Checks the decode path,
    /// KV reuse across steps, and positions.
    #[test]
    fn prefill_then_decode_matches_full_prefill() {
        let (n, k) = (13, 5);
        let p = prompt(n, 3);
        let table: Vec<u32> = vec![6, 1, 9, 3]; // shuffled; covers 16 ≥ 13

        let (mut m1, mut kv1) = tiny(small_engine_config());
        let full = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 0,
                num_new: n,
                block_table: &table,
            }],
            4,
        )
        .unwrap();
        let want = m1.prefill(&mut kv1, &p, &full, true).unwrap();

        let (mut m2, mut kv2) = tiny(small_engine_config());
        let head = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 0,
                num_new: n - k,
                block_table: &table,
            }],
            4,
        )
        .unwrap();
        let mut out = m2.prefill(&mut kv2, &p[..n - k], &head, true).unwrap();
        for pos in n - k..n {
            let meta = DecodeMeta::new(
                &[DecodeSeq {
                    context_len: pos,
                    block_table: &table,
                }],
                4,
            )
            .unwrap();
            out = m2.decode(&mut kv2, &p[pos..pos + 1], &meta, true).unwrap();
        }
        assert_eq!(out.tokens, want.tokens);
        let d = max_rel_diff(out.logits.as_ref().unwrap(), want.logits.as_ref().unwrap());
        assert!(d < 1e-5, "max rel diff {d}");
    }

    /// Prefill split into a cached context + new tokens (`context_len > 0`,
    /// the M4 prefix-hit path) matches one full prefill.
    #[test]
    fn prefill_with_context_matches_full_prefill() {
        let n = 11;
        let p = prompt(n, 5);
        let table: Vec<u32> = vec![2, 0, 7];
        let (mut m1, mut kv1) = tiny(small_engine_config());
        let full = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 0,
                num_new: n,
                block_table: &table,
            }],
            4,
        )
        .unwrap();
        let want = m1.prefill(&mut kv1, &p, &full, true).unwrap();

        let (mut m2, mut kv2) = tiny(small_engine_config());
        let a = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 0,
                num_new: 6,
                block_table: &table,
            }],
            4,
        )
        .unwrap();
        m2.prefill(&mut kv2, &p[..6], &a, false).unwrap();
        let b = PrefillMeta::new(
            &[PrefillSeq {
                context_len: 6,
                num_new: 5,
                block_table: &table,
            }],
            4,
        )
        .unwrap();
        let got = m2.prefill(&mut kv2, &p[6..], &b, true).unwrap();
        assert_eq!(got.tokens, want.tokens);
        assert!(max_rel_diff(got.logits.as_ref().unwrap(), want.logits.as_ref().unwrap()) < 1e-5);
    }

    /// Batch invariance (CLAUDE.md): two sequences prefilled together give the
    /// same logits as each alone. Bitwise, because no op mixes rows.
    #[test]
    fn batched_prefill_is_batch_invariant() {
        let (pa, pb) = (prompt(7, 1), prompt(10, 2));
        let (ta, tb): (Vec<u32>, Vec<u32>) = (vec![3, 5], vec![0, 8, 1]);
        let alone = |p: &[u32], t: &[u32]| {
            let (mut m, mut kv) = tiny(small_engine_config());
            let meta = PrefillMeta::new(
                &[PrefillSeq {
                    context_len: 0,
                    num_new: p.len(),
                    block_table: t,
                }],
                4,
            )
            .unwrap();
            m.prefill(&mut kv, p, &meta, true).unwrap()
        };
        let (a, b) = (alone(&pa, &ta), alone(&pb, &tb));

        let (mut m, mut kv) = tiny(small_engine_config());
        let meta = PrefillMeta::new(
            &[
                PrefillSeq {
                    context_len: 0,
                    num_new: pa.len(),
                    block_table: &ta,
                },
                PrefillSeq {
                    context_len: 0,
                    num_new: pb.len(),
                    block_table: &tb,
                },
            ],
            4,
        )
        .unwrap();
        let tokens: Vec<u32> = pa.iter().chain(&pb).copied().collect();
        let both = m.prefill(&mut kv, &tokens, &meta, true).unwrap();
        assert_eq!(both.tokens, [a.tokens[0], b.tokens[0]]);
        let l = both.logits.unwrap();
        let v = 37;
        assert_eq!(&l[..v], &a.logits.unwrap()[..]);
        assert_eq!(&l[v..], &b.logits.unwrap()[..]);
    }
}
