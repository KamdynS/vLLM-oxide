---
description: Open the current changes in plannotator for the owner to annotate, then act on the feedback
argument-hint: "[--base <ref>] [PR_URL]"
---

Open a plannotator code review: run `plannotator review --json $ARGUMENTS` with Bash in the background (it blocks until the owner submits in the browser, which can take longer than a foreground command allows). Tell the owner the review is open, then wait for it to finish.

When it returns:

1. Read every annotation. Treat each one as a review comment from the owner.
2. For each one, say what you'll change, or explain why you think no change is needed. Don't silently skip any.
3. Make the changes, then offer to open the review again.
4. If the owner approved a diff touching core code (kernels, block manager, scheduler, prefix cache, sampler, router policy) with no annotations, ask one comprehension question about it before moving on (see `docs/process.md`, "Staying honest").
