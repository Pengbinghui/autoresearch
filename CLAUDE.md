# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Structure

This repo contains multiple autoresearch-style projects in subfolders. Each follows the same pattern: an agent autonomously modifies code, runs a fixed-budget experiment, evaluates, keeps or discards, and repeats.

- **`gpu-pretraining/`** — Karpathy's original autoresearch. Trains a GPT language model. Requires NVIDIA GPU.
- **`mnist-search/`** — MNIST digit classifier search. CPU only.

Each project has its own `pyproject.toml`, `program.md`, and `CLAUDE.md`. Read the project's `program.md` and `CLAUDE.md` for setup and usage details.

## Working on a Project

```bash
cd <project>/
uv sync          # install dependencies
```

Then follow the project's `program.md` for instructions.

## Experiment Branches

Use branch naming: `autoresearch/<project>-<tag>` (e.g. `autoresearch/mnist-mar9`, `autoresearch/gpu-mar9`).
