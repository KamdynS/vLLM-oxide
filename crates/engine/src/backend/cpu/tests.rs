//! CPU ops against naive f64 reference implementations, through the public
//! `arena::Scope` API. Shapes are deliberately odd: not multiples of the dot
//! product's 16 lanes or matmul's 16-row tasks.

use super::*;
use crate::arena::{Arena, Scope, Tensor};
use crate::backend::{DecodeSeq, PrefillSeq};
use crate::testutil::{max_rel_diff, rand_vec, rand_weight, tiny_config};

const CAP: usize = 1 << 16;

fn setup() -> (CpuBackend, Arena) {
    let mut b = CpuBackend::new(&tiny_config(), 64);
    b.alloc_workspace(CAP).unwrap();
    let arena = Arena::new(&b);
    (b, arena)
}

/// Put host data into the arena: upload it as an embedding table and look up
/// every row. (There is deliberately no general host→arena copy op.)
fn tensor<'s>(s: &Scope<'s, CpuBackend>, data: &[f32], rows: usize, cols: usize) -> Tensor<'s> {
    let w = Weight::new(vec![rows, cols], data.to_vec()).unwrap();
    let ids: Vec<u32> = (0..rows as u32).collect();
    s.embed(&w, &ids).unwrap()
}

#[test]
fn dot_matches_naive_for_odd_lengths() {
    for n in [0, 1, 15, 16, 17, 33, 2048] {
        let (a, b) = (rand_vec(1, n), rand_vec(2, n));
        let want: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!((dot(&a, &b) as f64 - want).abs() < 1e-4, "n={n}");
    }
}

#[test]
fn matmul_matches_naive() {
    let (b, mut a) = setup();
    // 37 output rows = two full 16-row tasks + a partial one; 21 inputs = one
    // 16-lane chunk + remainder.
    let (t, n_in, n_out) = (7, 21, 37);
    let x = rand_vec(3, t * n_in);
    let w = rand_weight(4, &[n_out, n_in], 1.0);
    let mut want = vec![0f32; t * n_out];
    for i in 0..t {
        for o in 0..n_out {
            want[i * n_out + o] = (0..n_in)
                .map(|k| x[i * n_in + k] as f64 * w.data()[o * n_in + k] as f64)
                .sum::<f64>() as f32;
        }
    }
    let got = a.scope(&b, |s| {
        let xt = tensor(s, &x, t, n_in);
        let y = s.matmul(&xt, &w).unwrap();
        assert_eq!(y.shape(), [t, n_out]);
        s.read(&y).unwrap()
    });
    assert!(max_rel_diff(&got, &want) < 1e-6);
}

#[test]
fn matmul_rejects_mismatched_shapes() {
    let (b, mut a) = setup();
    let w = rand_weight(4, &[5, 9], 1.0);
    a.scope(&b, |s| {
        let x = tensor(s, &rand_vec(1, 16), 2, 8);
        assert!(matches!(
            s.matmul(&x, &w),
            Err(EngineError::Shape { op: "matmul", .. })
        ));
    });
}

#[test]
fn rmsnorm_matches_naive() {
    let (b, mut a) = setup();
    let (t, h) = (5, 19);
    let x = rand_vec(5, t * h);
    let w = rand_weight(6, &[h], 1.0);
    let eps = 1e-5;
    let mut want = vec![0f32; t * h];
    for i in 0..t {
        let r = &x[i * h..][..h];
        let ms = r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / h as f64;
        for j in 0..h {
            want[i * h + j] = (w.data()[j] as f64 * r[j] as f64 / (ms + eps).sqrt()) as f32;
        }
    }
    let got = a.scope(&b, |s| {
        let xt = tensor(s, &x, t, h);
        let y = s.rmsnorm(&xt, &w, eps as f32).unwrap();
        s.read(&y).unwrap()
    });
    assert!(max_rel_diff(&got, &want) < 1e-6);
}

#[test]
fn silu_mul_add_gather() {
    let (b, mut a) = setup();
    let (t, c) = (3, 7);
    let g = rand_vec(7, t * c);
    let u = rand_vec(8, t * c);
    a.scope(&b, |s| {
        let mut gt = tensor(s, &g, t, c);
        let ut = tensor(s, &u, t, c);
        s.silu_mul(&mut gt, &ut).unwrap();
        let got = s.read(&gt).unwrap();
        for i in 0..t * c {
            let x = g[i] as f64;
            let want = x / (1.0 + (-x).exp()) * u[i] as f64;
            assert!((got[i] as f64 - want).abs() < 1e-6);
        }

        s.add(&mut gt, &ut).unwrap();
        let got2 = s.read(&gt).unwrap();
        for i in 0..t * c {
            assert_eq!(got2[i], got[i] + u[i]);
        }

        let picked = s.gather_rows(&ut, &[2, 0]).unwrap();
        let p = s.read(&picked).unwrap();
        assert_eq!(&p[..c], &u[2 * c..3 * c]);
        assert_eq!(&p[c..], &u[..c]);
        assert!(s.gather_rows(&ut, &[3]).is_err());
    });
}

#[test]
fn argmax_first_max_wins_and_nan_errors() {
    let (b, mut a) = setup();
    a.scope(&b, |s| {
        let x = tensor(
            s,
            &[
                1.0, 5.0, 5.0, -1.0, 0.0, 0.0, 0.0, 0.0, -3.0, -2.0, -2.0, -9.0,
            ],
            3,
            4,
        );
        assert_eq!(s.argmax(&x).unwrap(), [1, 0, 1]);
        let n = tensor(s, &[0.0, f32::NAN], 1, 2);
        assert_eq!(s.argmax(&n).unwrap_err(), EngineError::NaN { op: "argmax" });
    });
}

#[test]
fn rope_rejects_positions_past_table() {
    let (b, mut a) = setup();
    a.scope(&b, |s| {
        let mut x = tensor(s, &rand_vec(1, 16), 1, 16);
        let e = s.rope(&mut x, &[64], 8).unwrap_err();
        assert_eq!(e, EngineError::PositionOutOfRange { pos: 64, max: 64 });
        s.rope(&mut x, &[63], 8).unwrap();
    });
}

// ---------------------------------------------------------------------------
// Paged attention vs naive contiguous attention.
// ---------------------------------------------------------------------------

const HEADS: usize = 6;
const KV_HEADS: usize = 2;
const HD: usize = 8;
const BS: usize = 4;
const NUM_BLOCKS: usize = 16;

fn kv_layout() -> KvLayout {
    KvLayout {
        num_layers: 2,
        num_blocks: NUM_BLOCKS,
        block_size: BS,
        kv_heads: KV_HEADS,
        head_dim: HD,
    }
}

/// One sequence's full K and V, contiguous: `[len, kv_heads * head_dim]`.
struct SeqKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Naive causal GQA attention in f64 for one query row at `pos`.
fn naive_attend(q: &[f32], kv: &SeqKv, pos: usize) -> Vec<f32> {
    let w = KV_HEADS * HD;
    let group = HEADS / KV_HEADS;
    let mut out = vec![0f32; HEADS * HD];
    for h in 0..HEADS {
        let g = h / group;
        let scores: Vec<f64> = (0..=pos)
            .map(|j| {
                (0..HD)
                    .map(|d| q[h * HD + d] as f64 * kv.k[j * w + g * HD + d] as f64)
                    .sum::<f64>()
                    / (HD as f64).sqrt()
            })
            .collect();
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = e.iter().sum();
        for d in 0..HD {
            out[h * HD + d] = (0..=pos)
                .map(|j| e[j] / z * kv.v[j * w + g * HD + d] as f64)
                .sum::<f64>() as f32;
        }
    }
    out
}

/// Write positions `range` of `kv` into `layer` of the pool via `write_kv`.
fn write_positions(
    s: &Scope<'_, CpuBackend>,
    pool: &mut CpuKvPool,
    layer: usize,
    kv: &SeqKv,
    table: &[u32],
    range: std::ops::Range<usize>,
) {
    let w = KV_HEADS * HD;
    let n = range.len();
    if n == 0 {
        return;
    }
    let k = tensor(s, &kv.k[range.start * w..range.end * w], n, w);
    let v = tensor(s, &kv.v[range.start * w..range.end * w], n, w);
    let slots: Vec<u32> = range
        .map(|p| crate::backend::slot_of(table, p, BS))
        .collect();
    s.write_kv(pool, layer, &k, &v, &slots).unwrap();
}

#[test]
fn paged_prefill_attention_matches_naive_with_shuffled_blocks() {
    let (b, mut a) = setup();
    let mut pool = b.alloc_kv_pool(kv_layout()).unwrap();
    let w = KV_HEADS * HD;
    let layer = 1; // not layer 0, so the layer index is exercised

    // Three sequences: (context_len, num_new). Context lengths and totals land
    // mid-block, on a block boundary, and at zero.
    let shape = [(5, 6), (0, 3), (8, 4)];
    // Non-identity, interleaved physical blocks: the identity table M0 uses
    // in practice would hide indexing bugs.
    let tables: [Vec<u32>; 3] = [vec![9, 2, 14], vec![7], vec![0, 11, 5]];
    let kvs: Vec<SeqKv> = shape
        .iter()
        .enumerate()
        .map(|(i, &(c, n))| SeqKv {
            k: rand_vec(100 + i as u64, (c + n) * w),
            v: rand_vec(200 + i as u64, (c + n) * w),
        })
        .collect();
    let seqs: Vec<PrefillSeq> = shape
        .iter()
        .zip(&tables)
        .map(|(&(c, n), t)| PrefillSeq {
            context_len: c,
            num_new: n,
            block_table: t,
        })
        .collect();
    let meta = PrefillMeta::new(&seqs, BS).unwrap();
    let t_rows = meta.num_rows();
    let q = rand_vec(300, t_rows * HEADS * HD);

    let got = a.scope(&b, |s| {
        for (i, &(c, n)) in shape.iter().enumerate() {
            write_positions(s, &mut pool, layer, &kvs[i], &tables[i], 0..c); // "earlier steps"
            write_positions(s, &mut pool, layer, &kvs[i], &tables[i], c..c + n); // this step
        }
        let qt = tensor(s, &q, t_rows, HEADS * HD);
        let out = s.prefill_attention(&qt, &pool, layer, &meta).unwrap();
        s.read(&out).unwrap()
    });

    let mut want = Vec::new();
    for (i, &(c, n)) in shape.iter().enumerate() {
        for j in 0..n {
            let row = meta.seq_row_starts[i] + j;
            want.extend(naive_attend(
                &q[row * HEADS * HD..][..HEADS * HD],
                &kvs[i],
                c + j,
            ));
        }
    }
    assert!(
        max_rel_diff(&got, &want) < 1e-5,
        "{}",
        max_rel_diff(&got, &want)
    );
}

#[test]
fn paged_decode_attention_matches_naive_with_shuffled_blocks() {
    let (b, mut a) = setup();
    let mut pool = b.alloc_kv_pool(kv_layout()).unwrap();
    let w = KV_HEADS * HD;
    // Context lengths: one key only, exactly one full block, spanning blocks.
    let ctx = [0usize, 3, 10];
    let tables: [Vec<u32>; 3] = [vec![15], vec![4], vec![12, 1, 8]];
    let kvs: Vec<SeqKv> = ctx
        .iter()
        .enumerate()
        .map(|(i, &c)| SeqKv {
            k: rand_vec(400 + i as u64, (c + 1) * w),
            v: rand_vec(500 + i as u64, (c + 1) * w),
        })
        .collect();
    let seqs: Vec<DecodeSeq> = ctx
        .iter()
        .zip(&tables)
        .map(|(&c, t)| DecodeSeq {
            context_len: c,
            block_table: t,
        })
        .collect();
    let meta = DecodeMeta::new(&seqs, BS).unwrap();
    let q = rand_vec(600, 3 * HEADS * HD);

    let got = a.scope(&b, |s| {
        for (i, &c) in ctx.iter().enumerate() {
            write_positions(s, &mut pool, 0, &kvs[i], &tables[i], 0..c + 1);
        }
        let qt = tensor(s, &q, 3, HEADS * HD);
        let out = s.decode_attention(&qt, &pool, 0, &meta).unwrap();
        s.read(&out).unwrap()
    });

    let want: Vec<f32> = (0..3)
        .flat_map(|i| naive_attend(&q[i * HEADS * HD..][..HEADS * HD], &kvs[i], ctx[i]))
        .collect();
    assert!(max_rel_diff(&got, &want) < 1e-5);
}

#[test]
fn out_of_range_blocks_and_slots_are_errors() {
    let (b, mut a) = setup();
    let mut pool = b.alloc_kv_pool(kv_layout()).unwrap();
    let bad = [NUM_BLOCKS as u32];
    let meta = DecodeMeta::new(
        &[DecodeSeq {
            context_len: 0,
            block_table: &bad,
        }],
        BS,
    )
    .unwrap();
    a.scope(&b, |s| {
        let q = tensor(s, &rand_vec(1, HEADS * HD), 1, HEADS * HD);
        assert!(matches!(
            s.decode_attention(&q, &pool, 0, &meta),
            Err(EngineError::BlockOutOfRange { .. })
        ));
        let k = tensor(s, &rand_vec(1, KV_HEADS * HD), 1, KV_HEADS * HD);
        let v = tensor(s, &rand_vec(2, KV_HEADS * HD), 1, KV_HEADS * HD);
        assert!(matches!(
            s.write_kv(&mut pool, 0, &k, &v, &[(NUM_BLOCKS * BS) as u32]),
            Err(EngineError::SlotOutOfRange { .. })
        ));
    });
}
