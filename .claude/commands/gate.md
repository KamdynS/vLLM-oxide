---
description: Run a milestone gate — check exit criteria, then grill
argument-hint: M0 | M1 | M2 | M3 | M4 | M5
---

Run the milestone gate for $ARGUMENTS.

1. Read the Milestones table in CLAUDE.md and the exit criteria for $ARGUMENTS in the project plan.
2. Check each criterion against the repo: run the relevant tests, check `docs/results.md` for the required measurements, and check `docs/decisions/` for ADRs this milestone should have produced. Report each criterion as met, not met, or unverifiable, with the evidence.
3. If any criterion is not met, stop and list what's missing. Do not grill yet.
4. If all are met, grill the owner across everything built so far, weighted toward this milestone's components. This is a comprehension check, not planning: do not use the `/grill-me` skill and give no recommended answers or hints. One question at a time; push for numbers; no notes allowed.
5. Append a session entry to `docs/grill-log.md`. Only if the grilling passes, update the milestone's status in CLAUDE.md to Done.
