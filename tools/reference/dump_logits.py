"""Dump HF transformers' greedy output for each prompt, as the M0 ground truth.

For every prompt in a JSONL file ({"id": ..., "text": ...}) this runs Llama in
float32 on the CPU with greedy decoding and writes <out>/<id>.safetensors with:

  prompt_ids      int32 [P]     tokenizer(text) ids, BOS included
  generated_ids   int32 [N]     greedy tokens, N <= max_new (stops after EOS)
  gaps            float32 [N]   top-1 minus top-2 logit at each step: small gaps
                                are near-ties where float noise can flip a token
  capture_steps   int32 [K]     generated-token indices whose logits are stored
  logits_<i>      float32 [V]   the logits that produced generated_ids[i]

Step 0 is the prefill's last position; step i > 0 is the decode step that
produced token i. f32 and eager attention on both sides, so the comparison is
about the implementation, not about dtype.

Run:
  nix-shell tools/reference/shell.nix --run \
    "python tools/reference/dump_logits.py --model models/Llama-3.2-1B \
     --prompts tools/reference/prompts.jsonl"
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Logits stored in full: the prefill and three decode steps spread over the
# 256-token run (early, middle, last). Full logits are 128,256 floats (0.5 MB)
# each, so storing all 256 steps would be 128 MB per prompt.
CAPTURE_STEPS = [0, 1, 64, 255]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="tools/reference/out")
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    eos = model.config.eos_token_id
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    for p in prompts:
        ids = tok(p["text"], return_tensors="pt").input_ids
        generated, gaps, captured = [], [], {}
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            for i in range(args.max_new):
                logits = out.logits[0, -1].float()
                top2 = torch.topk(logits, 2).values
                gaps.append((top2[0] - top2[1]).item())
                nxt = int(torch.argmax(logits))
                generated.append(nxt)
                if i in CAPTURE_STEPS:
                    captured[f"logits_{i}"] = logits.clone()
                if nxt == eos or i == args.max_new - 1:
                    break
                out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
                past = out.past_key_values

        tensors = {
            "prompt_ids": ids[0].to(torch.int32).contiguous(),
            "generated_ids": torch.tensor(generated, dtype=torch.int32),
            "gaps": torch.tensor(gaps, dtype=torch.float32),
            "capture_steps": torch.tensor(
                sorted(int(k.split("_")[1]) for k in captured), dtype=torch.int32
            ),
            **captured,
        }
        save_file(tensors, str(out_dir / f"{p['id']}.safetensors"))
        print(
            f"{p['id']}: prompt {ids.shape[1]} tokens, generated {len(generated)}, "
            f"min gap {min(gaps):.4f}"
        )


if __name__ == "__main__":
    main()
