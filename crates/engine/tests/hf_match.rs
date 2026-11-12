//! The M0 gate: greedy output matches HF transformers for 256 tokens on 20
//! prompts. Reference dumps come from `tools/reference/dump_logits.py`.
//!
//! Needs the model and the dumps, so it's `#[ignore]`d:
//!   nix-shell tools/reference/shell.nix --run "python tools/reference/dump_logits.py \
//!     --model models/Llama-3.2-1B --prompts tools/reference/prompts.jsonl"
//!   cargo test -p engine --release --test hf_match -- --ignored --nocapture

use std::path::PathBuf;

use engine::{CpuBackend, EngineConfig, Generator, LLAMA_3_2_1B, Llama, ModelWeights, Tokenizer};
use safetensors::{Dtype, SafeTensors};

/// Max |logit difference| allowed at the captured steps, in logit units.
/// Measured 2026-11-07 (docs/results.md): worst 5.4e-5 over 20 prompts × 4
/// steps, typical 1–2e-5. Both sides are f32; the difference is summation
/// order (our 16-lane dot vs torch's kernels) compounded over 16 layers. 2e-4
/// is ~4× headroom and still 4.5× below HF's smallest top-1/top-2 gap (9e-4,
/// p01), so a diff within tolerance can't by itself flip a greedy token.
const LOGIT_ABS_TOL: f32 = 2e-4;

const MAX_NEW: usize = 256;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_dir() -> PathBuf {
    std::env::var_os("LLAMA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo().join("models/Llama-3.2-1B"))
}

struct Reference {
    id: String,
    text: String,
    prompt_ids: Vec<u32>,
    generated: Vec<u32>,
    gaps: Vec<f32>,
    logits: Vec<(usize, Vec<f32>)>,
}

fn i32s(st: &SafeTensors, name: &str) -> Vec<i32> {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
    assert_eq!(t.dtype(), Dtype::I32, "{name}");
    t.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

fn f32s(st: &SafeTensors, name: &str) -> Vec<f32> {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
    assert_eq!(t.dtype(), Dtype::F32, "{name}");
    t.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn load_references() -> Vec<Reference> {
    let prompts = std::fs::read_to_string(repo().join("tools/reference/prompts.jsonl")).unwrap();
    prompts
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let id = v["id"].as_str().unwrap().to_string();
            let text = v["text"].as_str().unwrap().to_string();
            let path = repo().join(format!("tools/reference/out/{id}.safetensors"));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("{}: {e} (run dump_logits.py)", path.display()));
            let st = SafeTensors::deserialize(&bytes).unwrap();
            let u = |v: Vec<i32>| v.into_iter().map(|x| x as u32).collect::<Vec<u32>>();
            let steps = i32s(&st, "capture_steps");
            Reference {
                prompt_ids: u(i32s(&st, "prompt_ids")),
                generated: u(i32s(&st, "generated_ids")),
                gaps: f32s(&st, "gaps"),
                logits: steps
                    .iter()
                    .map(|&s| (s as usize, f32s(&st, &format!("logits_{s}"))))
                    .collect(),
                id,
                text,
            }
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
#[ignore = "needs models/Llama-3.2-1B and tools/reference/out (run dump_logits.py)"]
fn greedy_matches_hf_for_256_tokens_on_20_prompts() {
    let refs = load_references();
    assert_eq!(refs.len(), 20);

    let dir = model_dir();
    let tok = Tokenizer::load(&dir).unwrap();
    let engine_config = EngineConfig::default();
    let weights = ModelWeights::load(&dir, &LLAMA_3_2_1B).unwrap();
    let backend = CpuBackend::new(&LLAMA_3_2_1B, engine_config.max_model_len);
    let mut g =
        Generator::new(Llama::new(backend, LLAMA_3_2_1B, engine_config, weights).unwrap()).unwrap();

    let mut failures = Vec::new();
    let mut worst_logit_diff = 0f32;
    for r in &refs {
        // Tokenizer: identical ids, BOS included.
        let ids = tok.encode(&r.text).unwrap();
        assert_eq!(ids, r.prompt_ids, "{}: tokenizer ids differ from HF", r.id);

        let steps: Vec<usize> = r.logits.iter().map(|(s, _)| *s).collect();
        let out = g
            .greedy(&ids, MAX_NEW, Some(LLAMA_3_2_1B.eos_token_id), &steps)
            .unwrap();

        let mut diffs = Vec::new();
        for ((s, want), (s2, got)) in r.logits.iter().zip(&out.captured) {
            assert_eq!(s, s2);
            let d = max_abs_diff(got, want);
            worst_logit_diff = worst_logit_diff.max(d);
            diffs.push(format!("step {s}: {d:.2e}"));
            if d > LOGIT_ABS_TOL {
                failures.push(format!(
                    "{}: logits at step {s} differ by {d:.3e} > {LOGIT_ABS_TOL:e}",
                    r.id
                ));
            }
        }

        let first_mismatch = out
            .tokens
            .iter()
            .zip(&r.generated)
            .position(|(a, b)| a != b);
        match first_mismatch {
            None if out.tokens.len() == r.generated.len() => {}
            None => failures.push(format!(
                "{}: lengths differ: ours {}, HF {}",
                r.id,
                out.tokens.len(),
                r.generated.len()
            )),
            Some(i) => failures.push(format!(
                "{}: first token mismatch at step {i}: ours {}, HF {} (HF top-1/top-2 gap there: {:.4})",
                r.id, out.tokens[i], r.generated[i], r.gaps[i]
            )),
        }
        println!(
            "{:<18} prompt {:>3} gen {:>3}  min gap {:.4}  {}",
            r.id,
            ids.len(),
            out.tokens.len(),
            r.gaps.iter().copied().fold(f32::INFINITY, f32::min),
            diffs.join(", ")
        );
    }
    println!("worst logit |diff| over all captured steps: {worst_logit_diff:.3e}");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
