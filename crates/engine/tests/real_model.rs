//! Checks against the real Llama 3.2 1B download. Needs the 2.5 GB files, so
//! these are `#[ignore]`d; run with `cargo test -p engine --release -- --ignored`.
//! Model dir: `$LLAMA_DIR`, default `<repo>/models/Llama-3.2-1B`.

use std::path::PathBuf;

use engine::{LLAMA_3_2_1B, ModelWeights};
use serde_json::{Value, json};

fn model_dir() -> PathBuf {
    std::env::var_os("LLAMA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Llama-3.2-1B")
        })
}

/// The hardcoded const was transcribed correctly, and the file describes the
/// architecture we implement. Checked once here, not at every startup.
#[test]
#[ignore = "needs models/Llama-3.2-1B (2.5 GB)"]
fn const_matches_downloaded_config() {
    let text = std::fs::read_to_string(model_dir().join("config.json")).unwrap();
    let file: Value = serde_json::from_str(&text).unwrap();
    let c = &LLAMA_3_2_1B;
    let rs = &c.rope_scaling;
    let expected = [
        ("hidden_size", json!(c.hidden_size)),
        ("intermediate_size", json!(c.intermediate_size)),
        ("num_hidden_layers", json!(c.num_hidden_layers)),
        ("num_attention_heads", json!(c.num_attention_heads)),
        ("num_key_value_heads", json!(c.num_key_value_heads)),
        ("head_dim", json!(c.head_dim())),
        ("vocab_size", json!(c.vocab_size)),
        ("max_position_embeddings", json!(c.max_position_embeddings)),
        ("rms_norm_eps", json!(c.rms_norm_eps)),
        ("rope_theta", json!(c.rope_theta)),
        ("bos_token_id", json!(c.bos_token_id)),
        ("eos_token_id", json!(c.eos_token_id)),
        // Architecture we implement but don't store as fields.
        ("model_type", json!("llama")),
        ("hidden_act", json!("silu")),
        ("tie_word_embeddings", json!(true)),
        ("attention_bias", json!(false)),
        ("mlp_bias", json!(false)),
    ];
    for (key, want) in expected {
        assert_eq!(file[key], want, "{key}");
    }
    let rope = &file["rope_scaling"];
    assert_eq!(rope["rope_type"], json!("llama3"));
    assert_eq!(rope["factor"], json!(rs.factor));
    assert_eq!(rope["low_freq_factor"], json!(rs.low_freq_factor));
    assert_eq!(rope["high_freq_factor"], json!(rs.high_freq_factor));
    assert_eq!(
        rope["original_max_position_embeddings"],
        json!(rs.original_max_position_embeddings)
    );
}

#[test]
#[ignore = "needs models/Llama-3.2-1B (2.5 GB)"]
fn loads_real_weights() {
    let w = ModelWeights::load(&model_dir(), &LLAMA_3_2_1B).unwrap();
    // 1 embed + 16 layers × 9 + 1 final norm.
    assert_eq!(w.named().len(), 146);
    // Owner's hand count, now checked against the actual file.
    assert_eq!(w.param_count(), 1_235_814_400);
    // Sanity: RMSNorm weights in a trained model are not all zero.
    assert!(w.final_norm.data().iter().any(|&x| x != 0.0));
}
