# Grill log

One entry per grilling session, newest first. Missed questions become tasks or ADRs before the next session.

### 2026-11-05 — design — M0 PR1 (config + weights)

- **Result:** Pass with follow-ups (the grill itself failed, not the owner)
- **Answered well:**
  - Hand-counted Llama 3.2 1B params: 1,235,814,400, missing only the 2,048-param final norm. KV: 64 KiB/token f32, 32 KiB bf16; ~61k tokens in 4 GB.
  - Picked Llama 3.2 1B (ADR 0002); strict config rejection; bf16→f32 at load; tied embedding stored once.
- **Missed or shaky:**
  - Peak RAM while loading: not answered, because the question was unclear (it said "the file" without saying which). Answer: ~2.5 GB file bytes + 4.9 GB f32 ≈ 7.4 GB.
  - Load-time estimate: "about a second". Checked against the measurement in results.md.
- **What went wrong with the grill:** four rounds and about an hour on plumbing (fixture paths, model dir, test mechanics, mmap) and none on architecture. One "question" had no question in it.
- **Follow-ups:**
  - [x] New rule: grill architecture only, teach then ask, max 4 per round; Claude lists plumbing as "Claude's defaults" (docs/process.md "What gets grilled").
  - [ ] Next PR (CPU forward pass + backend trait) is the first real architecture grill under the new rule.

## Template

### YYYY-MM-DD — <scope: design | code | gate M?> — <component>

- **Result:** Pass | Pass with follow-ups | Fail
- **Answered well:**
  -
- **Missed or shaky:**
  - Question — what I said — what the right answer is
- **Follow-ups:**
  - [ ] Task or ADR to write
