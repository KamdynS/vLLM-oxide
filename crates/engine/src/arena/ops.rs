//! The ops: the safe API the model is written in, and the one place the
//! backend's `unsafe` op contract is kept.
//!
//! Every op does three things and no math (the math is in the backend):
//! 1. check shapes (contract promise 3);
//! 2. `alloc` its output from the arena (promises 1 and 2: in bounds, and
//!    fresh, so it overlaps no live tensor);
//! 3. call the backend's `unsafe` op.
//!
//! This is a child module of `arena` so it can use `Scope`'s private `alloc`
//! and `region` without making them visible to the rest of the crate: code
//! outside `arena` can't get a region without going through these checks.

use super::{Scope, Tensor};
use crate::backend::{Backend, DecodeMeta, EngineError, PrefillMeta, Region, Result};

impl<'scope, B: Backend> Scope<'scope, B> {
    // SAFETY, shared by every `unsafe` backend call below (the contract is on
    // `backend::Backend`):
    // 1. In bounds: every region came from `alloc` on this arena (checked by
    //    `region`), and `alloc` never goes past `capacity`, which `Arena::new`
    //    read from the backend's workspace size.
    // 2. No overlap: outputs are fresh `alloc`s above `top`, so they overlap no
    //    live tensor. In-place ops take their in-out tensor as `&mut Tensor`,
    //    so the borrow checker stops it also being passed as an input, and
    //    since `Tensor` isn't `Clone` there is no second handle to that region.
    // 3. Shapes: checked in each op before the call.

    /// tokens `[T]` → `[T, hidden]`.
    pub fn embed(&self, table: &B::Weight, tokens: &[u32]) -> Result<Tensor<'scope>> {
        self.op("embed", || {
            let [vocab, hidden] = weight_2d(self.backend, table, "embed")?;
            if let Some(&token) = tokens.iter().find(|&&t| t as usize >= vocab) {
                return Err(shape_err(
                    "embed",
                    format!("token {token} >= vocab {vocab}"),
                ));
            }
            let out = self.alloc(tokens.len(), hidden)?;
            // SAFETY: see the block comment above.
            unsafe { self.backend.embed(table, tokens, out.region)? };
            Ok(out)
        })
    }

    /// `[T, H]` → `[T, H]`.
    pub fn rmsnorm(&self, x: &Tensor<'_>, weight: &B::Weight, eps: f32) -> Result<Tensor<'scope>> {
        self.op("rmsnorm", || {
            let x = self.region(x)?;
            let weight_shape = self.backend.weight_shape(weight);
            if weight_shape != [x.cols] {
                return Err(shape_err(
                    "rmsnorm",
                    format!("x {x:?}, weight {weight_shape:?}"),
                ));
            }
            let out = self.alloc(x.rows, x.cols)?;
            // SAFETY: see the block comment above.
            unsafe { self.backend.rmsnorm(x, weight, eps, out.region)? };
            Ok(out)
        })
    }

    /// x `[T, in]`, weight `[out, in]` → `[T, out]`.
    pub fn matmul(&self, x: &Tensor<'_>, weight: &B::Weight) -> Result<Tensor<'scope>> {
        self.op("matmul", || {
            let x = self.region(x)?;
            let [out_features, in_features] = weight_2d(self.backend, weight, "matmul")?;
            if in_features != x.cols {
                return Err(shape_err(
                    "matmul",
                    format!("x {x:?}, weight [{out_features}, {in_features}]"),
                ));
            }
            let out = self.alloc(x.rows, out_features)?;
            // SAFETY: see the block comment above.
            unsafe { self.backend.matmul(x, weight, out.region)? };
            Ok(out)
        })
    }

    /// In place. x `[T, heads * head_dim]`, positions `[T]`.
    pub fn rope(&self, x: &mut Tensor<'_>, positions: &[u32], head_dim: usize) -> Result<()> {
        self.op("rope", || {
            let x = self.region(x)?;
            let shape_ok = positions.len() == x.rows
                && head_dim > 0
                && head_dim.is_multiple_of(2)
                && x.cols.is_multiple_of(head_dim);
            if !shape_ok {
                let msg = format!(
                    "x {x:?}, {} positions, head_dim {head_dim}",
                    positions.len()
                );
                return Err(shape_err("rope", msg));
            }
            // SAFETY: see the block comment above.
            unsafe { self.backend.rope(x, positions, head_dim) }
        })
    }

    /// keys, values `[T, kv_heads * head_dim]`, slots `[T]`.
    pub fn write_kv(
        &self,
        pool: &mut B::KvPool,
        layer: usize,
        keys: &Tensor<'_>,
        values: &Tensor<'_>,
        slots: &[u32],
    ) -> Result<()> {
        self.op("write_kv", || {
            let keys = self.region(keys)?;
            let values = self.region(values)?;
            let layout = self.backend.kv_layout(pool);
            let slot_width = layout.kv_heads * layout.head_dim;
            let shape_ok = keys.rows == slots.len()
                && values.rows == slots.len()
                && keys.cols == slot_width
                && values.cols == slot_width
                && layer < layout.num_layers;
            if !shape_ok {
                let msg = format!(
                    "keys {keys:?}, values {values:?}, {} slots, layer {layer}, pool {layout:?}",
                    slots.len()
                );
                return Err(shape_err("write_kv", msg));
            }
            // SAFETY: see the block comment above.
            unsafe { self.backend.write_kv(pool, layer, keys, values, slots) }
        })
    }

    /// queries `[T, heads * head_dim]` → `[T, heads * head_dim]`.
    pub fn prefill_attention(
        &self,
        queries: &Tensor<'_>,
        pool: &B::KvPool,
        layer: usize,
        meta: &PrefillMeta,
    ) -> Result<Tensor<'scope>> {
        self.op("prefill_attention", || {
            let queries = self.region(queries)?;
            let rows = meta.num_rows();
            self.check_attention(
                "prefill_attention",
                queries,
                pool,
                layer,
                rows,
                meta.block_size,
            )?;
            let out = self.alloc(queries.rows, queries.cols)?;
            // SAFETY: see the block comment above.
            unsafe {
                self.backend
                    .prefill_attention(queries, pool, layer, meta, out.region)?
            };
            Ok(out)
        })
    }

    /// queries `[S, heads * head_dim]` → `[S, heads * head_dim]`.
    pub fn decode_attention(
        &self,
        queries: &Tensor<'_>,
        pool: &B::KvPool,
        layer: usize,
        meta: &DecodeMeta,
    ) -> Result<Tensor<'scope>> {
        self.op("decode_attention", || {
            let queries = self.region(queries)?;
            let rows = meta.num_seqs();
            self.check_attention(
                "decode_attention",
                queries,
                pool,
                layer,
                rows,
                meta.block_size,
            )?;
            let out = self.alloc(queries.rows, queries.cols)?;
            // SAFETY: see the block comment above.
            unsafe {
                self.backend
                    .decode_attention(queries, pool, layer, meta, out.region)?
            };
            Ok(out)
        })
    }

    fn check_attention(
        &self,
        op: &'static str,
        queries: Region,
        pool: &B::KvPool,
        layer: usize,
        meta_rows: usize,
        block_size: usize,
    ) -> Result<()> {
        let layout = self.backend.kv_layout(pool);
        let shape_ok = queries.rows == meta_rows
            && queries.cols.is_multiple_of(layout.head_dim)
            && (queries.cols / layout.head_dim).is_multiple_of(layout.kv_heads)
            && layer < layout.num_layers
            && block_size == layout.block_size;
        if !shape_ok {
            let msg = format!(
                "queries {queries:?}, {meta_rows} rows in metadata, block size {block_size}, \
                 layer {layer}, pool {layout:?}"
            );
            return Err(shape_err(op, msg));
        }
        Ok(())
    }

    /// In place: `gate = silu(gate) * up`. Both `[T, intermediate]`.
    pub fn silu_mul(&self, gate: &mut Tensor<'_>, up: &Tensor<'_>) -> Result<()> {
        self.op("silu_mul", || {
            let gate = self.region(gate)?;
            let up = self.region(up)?;
            expect_same_shape("silu_mul", gate, up)?;
            // SAFETY: see the block comment above.
            unsafe { self.backend.silu_mul(gate, up) }
        })
    }

    /// In place: `sum += x`. Both `[T, H]`.
    pub fn add(&self, sum: &mut Tensor<'_>, x: &Tensor<'_>) -> Result<()> {
        self.op("add", || {
            let sum = self.region(sum)?;
            let x = self.region(x)?;
            expect_same_shape("add", sum, x)?;
            // SAFETY: see the block comment above.
            unsafe { self.backend.add(sum, x) }
        })
    }

    /// x `[T, H]`, rows `[S]` → `[S, H]`.
    pub fn gather_rows(&self, x: &Tensor<'_>, rows: &[u32]) -> Result<Tensor<'scope>> {
        self.op("gather_rows", || {
            let x = self.region(x)?;
            if let Some(&row) = rows.iter().find(|&&r| r as usize >= x.rows) {
                return Err(shape_err("gather_rows", format!("row {row} of {x:?}")));
            }
            let out = self.alloc(rows.len(), x.cols)?;
            // SAFETY: see the block comment above.
            unsafe { self.backend.gather_rows(x, rows, out.region)? };
            Ok(out)
        })
    }

    /// Greedy token per row: `[S, vocab]` → `[S]`. The one device→host sync
    /// per step.
    pub fn argmax(&self, logits: &Tensor<'_>) -> Result<Vec<u32>> {
        self.op("argmax", || {
            let logits = self.region(logits)?;
            // SAFETY: see the block comment above.
            unsafe { self.backend.argmax(logits) }
        })
    }

    /// Copy to the host. Debug and tests only.
    pub fn read(&self, x: &Tensor<'_>) -> Result<Vec<f32>> {
        let x = self.region(x)?;
        // SAFETY: see the block comment above.
        unsafe { self.backend.read(x) }
    }
}

fn shape_err(op: &'static str, msg: String) -> EngineError {
    EngineError::Shape { op, msg }
}

fn expect_same_shape(op: &'static str, a: Region, b: Region) -> Result<()> {
    if (a.rows, a.cols) != (b.rows, b.cols) {
        return Err(shape_err(op, format!("{a:?} vs {b:?}")));
    }
    Ok(())
}

fn weight_2d<B: Backend>(backend: &B, weight: &B::Weight, op: &'static str) -> Result<[usize; 2]> {
    match *backend.weight_shape(weight) {
        [rows, cols] => Ok([rows, cols]),
        ref shape => Err(shape_err(op, format!("weight must be 2-D, is {shape:?}"))),
    }
}
