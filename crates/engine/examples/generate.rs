//! Greedy generation from a prompt, with timing.
//!
//!   cargo run --release -p engine --example generate -- [model_dir] "prompt" [max_new] [--profile]
//!
//! Logs prefill tok/s (prompt tokens / prefill time) and decode tok/s
//! (generated tokens after the first / decode time). `--profile` adds per-op
//! time totals (opt-in, grill Q8).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use engine::{CpuBackend, EngineConfig, Generator, LLAMA_3_2_1B, Llama, ModelWeights, Tokenizer};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let profile = args.iter().any(|a| a == "--profile");
    args.retain(|a| a != "--profile");
    let dir = PathBuf::from(args.first().map_or("models/Llama-3.2-1B", String::as_str));
    let prompt = args
        .get(1)
        .map_or("The capital of France is", String::as_str);
    let max_new: usize = args
        .get(2)
        .map_or(Ok(64), |s| s.parse())
        .context("max_new")?;

    let t = Instant::now();
    let tok = Tokenizer::load(&dir)?;
    let weights = ModelWeights::load(&dir, &LLAMA_3_2_1B)?;
    let engine_config = EngineConfig::default();
    let backend = CpuBackend::new(&LLAMA_3_2_1B, engine_config.max_model_len);
    let mut model = Llama::new(backend, LLAMA_3_2_1B, engine_config, weights)?;
    if profile {
        model.enable_profiling();
    }
    let mut g = Generator::new(model)?;
    tracing::info!(secs = t.elapsed().as_secs_f64(), "loaded");

    let ids = tok.encode(prompt)?;
    let out = g.greedy(&ids, max_new, Some(LLAMA_3_2_1B.eos_token_id), &[])?;
    let prefill_s = out.prefill_time.as_secs_f64();
    let decode_s = out.decode_time.as_secs_f64();
    let decoded = out.tokens.len().saturating_sub(1);
    tracing::info!(
        prompt_tokens = ids.len(),
        generated = out.tokens.len(),
        prefill_ms = prefill_s * 1e3,
        prefill_tok_s = ids.len() as f64 / prefill_s,
        decode_tok_s = decoded as f64 / decode_s,
        ms_per_decode_token = decode_s * 1e3 / decoded.max(1) as f64,
        arena_high_water_mb = (g.model().arena().high_water() * 4) as f64 / 1e6,
        "done"
    );
    if let Some(p) = g.model().profile() {
        let mut ops: Vec<_> = p.into_iter().collect();
        ops.sort_by_key(|(_, s)| std::cmp::Reverse(s.total));
        for (name, s) in ops {
            tracing::info!(
                op = name,
                calls = s.calls,
                total_ms = s.total.as_secs_f64() * 1e3,
                "profile"
            );
        }
    }
    let text = tok.decode(&out.tokens)?;
    tracing::info!("{prompt}{text}");
    Ok(())
}
