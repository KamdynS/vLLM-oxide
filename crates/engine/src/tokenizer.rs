//! Text ↔ token ids, via the HF `tokenizers` crate reading `tokenizer.json`.
//!
//! Encoding adds the special tokens the file's post-processor defines, which
//! for Llama 3.2 base is a leading BOS (128000). That's what HF's
//! `AutoTokenizer(prompt)` does too, and `tests/hf_match.rs` checks the ids
//! are identical. Base model: no chat template.

use std::path::Path;

#[derive(Debug, thiserror::Error)]
#[error("tokenizer: {0}")]
pub struct TokenizerError(String);

pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    /// Load `<model_dir>/tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self, TokenizerError> {
        let path = model_dir.join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&path)
            .map(Tokenizer)
            .map_err(|e| TokenizerError(format!("{}: {e}", path.display())))
    }

    /// Token ids for `text`, with BOS prepended.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.0
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| TokenizerError(e.to_string()))
    }

    /// Text for `ids`, skipping special tokens (BOS, EOS).
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.0
            .decode(ids, true)
            .map_err(|e| TokenizerError(e.to_string()))
    }
}
