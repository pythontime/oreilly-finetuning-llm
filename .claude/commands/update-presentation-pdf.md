---
name: update-presentation-pdf
description: Workflow command scaffold for update-presentation-pdf in oreilly-finetuning-llm.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /update-presentation-pdf

Use this workflow when working on **update-presentation-pdf** in `oreilly-finetuning-llm`.

## Goal

Updates or adds presentation PDF files summarizing finetuning progress or results.

## Common Files

- `*Finetuning.pdf`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or generate new PDF presentation file(s) (e.g., YYYY-MM-DD Finetuning.pdf).
- Commit the new or updated PDF(s).

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.