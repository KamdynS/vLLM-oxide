---
description: Make me explain a file or function back, line by line, before it merges
argument-hint: <path or path::function>
---

Review $ARGUMENTS with me as a pre-merge check.

1. Read it. Don't summarize it for me.
2. Pick the lines most likely to hide a misunderstanding: `unsafe` blocks, constants, index math, sync points, error paths, anything touching KV block ownership.
3. Ask me to explain each one, one at a time. Push for the invariant, the shape, the bytes moved, or the failure mode.
4. After I answer, tell me plainly what I got wrong or left out.
5. List anything that should become a `// SAFETY:` comment, a `TODO(measure)`, a test, or an ADR.
