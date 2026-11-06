//! Load Llama 3.2 1B and report how long it took.
//!
//! Peak RSS: `nix shell nixpkgs#time -c time -v target/release/examples/load_weights [model_dir]`.
//! Cold cache: evict the file first with `dd if=<model_dir>/model.safetensors iflag=nocache count=0`.

use std::path::PathBuf;
use std::time::Instant;

use engine::{LLAMA_3_2_1B, ModelWeights};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Llama-3.2-1B"));

    let t = Instant::now();
    let w = ModelWeights::load(&dir, &LLAMA_3_2_1B)?;
    let secs = t.elapsed().as_secs_f64();
    tracing::info!(
        params = w.param_count(),
        f32_gb = (w.param_count() * 4) as f64 / 1e9,
        secs,
        "loaded weights"
    );
    Ok(())
}
