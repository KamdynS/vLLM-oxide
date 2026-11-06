# M0 PR1: fix the grill, then config + weight loading

## Context

The first grill for M0 went badly. It spent an hour on plumbing (fixture locations, file paths, test mechanics) and none on architecture. The owner wants to be part of the architecture, not to rubber-stamp minutiae. So this PR does two things:

1. **Fixes the process** so grills only ask architecture questions, and Claude decides plumbing without asking.
2. **Ships the first code**: the workspace, a typed `ModelConfig`, and loading the Llama 3.2 1B weights into f32. This is plumbing, so it runs as a "mechanical" PR (`process.md` allows skipping the grill and `/explain` for these). The owner reviews the diff in plannotator.

The next PR (CPU forward pass + backend trait) is real architecture and gets a proper grill under the new rules.

## Part A: process fix (applied first)

_Owner reviewed in plannotator (2026-11-05): approved the architecture-only rule, Claude's defaults, and teach-then-ask. No annotations on Part B; under the new rule, silence on plumbing means accepted._

**`docs/process.md`, step 2 and a new "What gets grilled" section:**
- **Grill only architecture.** A question qualifies only if it changes serving behavior, performance or memory, or an interface another component depends on, *or* a senior inference engineer would plausibly ask it in an interview. Examples: backend trait boundary, KV layout, who owns memory, prefill vs decode paths, sync points, scheduling policy.
- **Claude decides plumbing without asking.** That covers file locations, dependencies, fixtures, crate housekeeping, test mechanics, naming and error types. These are listed in the plan under **"Claude's defaults"**, one line each with a reason. The owner strikes or changes any of them in plannotator; silence means accepted.
- **Testing is not a grill topic.** CLAUDE.md already sets the bar, and Claude just meets it.
- **Teach, then ask.** Every question opens with 2–4 plain-English sentences: what the thing is and why it matters for serving. Then it asks one real question ending in "?". No jargon without a definition the first time.
- **At most 4 questions per round**, ordered by how much the answer shapes the system.
- **A plumbing-only PR skips the grill** and says so in the PR.

**`CLAUDE.md` Grilling section:** a one-line pointer to the rule above.

**Memory:** a feedback memory, "grills are for architecture only; Claude decides plumbing; teach before asking".

**`docs/grill-log.md`:** an entry for 2026-11-05 recording what went wrong and the rule change. The owner's estimate goes in "answered well": the param count of 1,235,814,400 (missed only the final norm, 2048) and KV of 64 KiB/token f32.

## Part B: the code

Branch `m0/01-config-and-weights` off a freshly pulled `main` (ADR 0001 merged as `b110143`).

### Decisions the owner made
- Model: **Llama 3.2 1B base**, recorded in ADR 0002 `docs/decisions/0002-model-llama-3.2-1b.md`. The reasons: Llama-family is the project's stated scope, 1B fits CPU f32 (4.9 GB), and 8B in f32 (32 GB) doesn't fit this box's 30 GB.
- PR scope: config + weights. Tokenizer, forward pass, Python and the backend trait are out.
- ~~Config is **strict**: parsed from config.json at runtime and rejected if it isn't our Llama variant.~~ **Changed in code review (2026-11-06):** the config is a hardcoded `const LLAMA_3_2_1B`. One model means the config is static data, and validating static data at runtime is pointless. Consistency is checked with compile-time asserts, and an ignored test checks the const against the downloaded config.json once. The weights stay validated at load because the safetensors file is the one real runtime input.
- `ModelConfig` has `const fn` methods: `head_dim`, `gqa_group`, `param_count`, `kv_bytes_per_token(dtype)`.
- Weights are converted **bf16 → f32 at load**.
- The **tied embedding is stored once** and used for both input and LM head.

### Claude's defaults (strike any in review)
- Root `Cargo.toml` workspace with `engine`, `kernels`, and `rust-version = "1.99"`. Delete the `add(2,2)` stubs. Don't create server/router/bench yet.
- Deps: `thiserror`, `safetensors`; `serde_json` is dev-only (the real-model config test). bf16→f32 is hand-written (`f32::from_bits((b as u32) << 16)`), so no `half` crate; it's one line and shows why the conversion is exact.
- Matrices keep HF's `[out_features, in_features]` row-major layout. vLLM does the same, and it's what decode's vector×matrix reads contiguously.
- `Weight { shape: Vec<usize>, data: Vec<f32> }`. Every framework's tensors carry their shape; this is the minimal version, and the forward-pass PR may replace it. `ModelWeights { embed, layers: Vec<LayerWeights>, final_norm }`, with `LayerWeights { attn_norm, wq, wk, wv, wo, mlp_norm, w_gate, w_up, w_down }`.
- Read `model.safetensors` with `std::fs::read`, not mmap. No `unsafe` in this PR. Peak RAM is about 2.5 GB file bytes + 4.9 GB f32 ≈ 7.4 GB.
- Errors: a `thiserror` `LoadError` that names the tensor or field and the expected vs actual shape. A missing, extra or misshaped tensor fails at load.
- Files: `crates/engine/src/{lib.rs, config.rs, weights.rs}`.
- Tests:
  - Unit tests live in `#[cfg(test)] mod tests` inside each module (changed in review; Rust convention).
  - Synthetic tiny safetensors (2 layers) built in-test, covering shapes, tied-embedding rule, missing or misshaped tensors, and exact bf16 round-trip.
  - A real-model test, `#[ignore]`, reading `LLAMA_DIR` (default `models/Llama-3.2-1B`). It checks all 146 tensor shapes and that `param_count() == 1_235_814_400`.
- Model location: `models/Llama-3.2-1B/`, already gitignored. The owner downloads it after accepting the license (command in the PR description).
- `crates/engine/examples/load_weights.rs`: loads the model and logs the elapsed time via `tracing`.

### Authorship
All code and docs: Claude. Owner: the estimates below, the ADR 0002 decision, and the plannotator review.

### Measurement → `docs/results.md`
- **Owner's estimate:** load time "about a second"; param count 1,235,814,400 (correct).
- **Measured:** cold-cache and warm-cache load time, and peak RSS via `/usr/bin/time -v cargo run --release --example load_weights`. The owner writes one sentence on the gap (disk-bound vs conversion-bound).

## Verification
- `cargo fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
- With the model downloaded: `cargo test -p engine -- --ignored` passes (shapes + exact param count).
- Run the example under `/usr/bin/time -v` and record the numbers.
- Save the approved plan as `docs/plans/0001-config-and-weights.md`. Add the `--ignored` command to CLAUDE.md Commands.

## Next PR preview (real grill)
The CPU forward pass and backend trait. Example questions: Does the backend trait own KV memory, or does the engine own it and hand out handles? Does M0's KV cache already use blocks, or a contiguous buffer per sequence that M2 replaces? Which operations cross the trait (whole layers, or individual ops like matmul/attention/rmsnorm), and what does that cost the CUDA backend later?
