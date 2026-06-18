---
name: add-or-update-notebooks-and-dependencies
description: Workflow command scaffold for add-or-update-notebooks-and-dependencies in oreilly-finetuning-llm.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-update-notebooks-and-dependencies

Use this workflow when working on **add-or-update-notebooks-and-dependencies** in `oreilly-finetuning-llm`.

## Goal

Adds or updates multiple Jupyter notebooks related to finetuning experiments, often alongside dependency files (pyproject.toml, requirements.txt).

## Common Files

- `*.ipynb`
- `pyproject.toml`
- `requirements.txt`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or add multiple .ipynb files related to model finetuning or data preparation.
- Update pyproject.toml and/or requirements.txt to reflect new or changed dependencies.
- Commit all changes together.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.