---
description: Draft a decision record by interviewing me, not by deciding for me
argument-hint: <decision, e.g. "KV block size">
---

Help me write an ADR for: $ARGUMENTS

1. Find the next free number in `docs/decisions/` and copy `0000-template.md` to `NNNN-<short-title>.md`.
2. Ask me, one question at a time: what forces the choice, which options I'm weighing, what I'd pick and why, and what evidence I have (measurements, back-of-envelope math).
3. Challenge weak evidence. If a claim can be measured, say what to measure and add a `TODO(M<n>): measure ...` instead of accepting a guess.
4. Write the ADR in my words. Don't add options or arguments I didn't raise without flagging them as your suggestions.
5. Update the open-decisions table in `docs/decisions/README.md` and, if the decision is settled, the "Settled so far" list in CLAUDE.md.
