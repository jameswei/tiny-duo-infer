# File Structure

This document is a navigation map for humans and agents. It intentionally stays
at directory level; architecture details belong in `docs/architecture.md`, phase
contracts belong in `docs/phases/`, and collaboration rules belong in
`docs/agent-guidelines.md`.

## Root

| Path | Purpose |
|---|---|
| `README.md` | Project overview, supported models, common commands, and roadmap status. |
| `AGENTS.md` | Short bootstrap for agents: what to read before changing code. |
| `pyproject.toml` | Python package metadata, runtime dependencies, and dev dependency groups. |
| `.github/workflows/` | Remote CI regression gates. |
| `models/` | Optional local symlinks to downloaded Hugging Face model snapshots; not required to be committed. |

## Source

| Path | Purpose |
|---|---|
| `tiny_duo_infer/` | Main Python package for the inference engine. Top-level modules include `engine.py`, `cli.py`, `generation.py`, `context_policy.py`, `profiling.py`, `quantization.py` (Phase 1.8 weight-only quantization config and packed weight representation), `prompt.py`, `sampling.py`, `cache.py`, and `config.py`. |
| `tiny_duo_infer/models/` | Model-family assembly, currently Llama and Qwen3. `models/base.py` defines the project `Module`, `Linear` (full-precision and quantized matmul dispatch), and `Embedding`. |
| `tiny_duo_infer/layers/` | Explicit learning-oriented neural-network layers: attention, RoPE, RMSNorm, FFN. |
| `tiny_duo_infer/weights/` | Safetensors loading, Hugging Face-to-project weight conversion, and Phase 1.8 in-memory weight-only quantization (`weights/quantizer.py`: eligible-Linear quantization step + `LinearWeightStats` memory accounting). |
| `tiny_duo_infer/tokenizer/` | Project tokenizer wrapper around `tokenizers`. |
| `tiny_duo_infer/serving/` | Single-request HTTP serving and worker lifecycle. Worker accepts an optional `QuantizationConfig` and loads the engine on the dedicated MLX GPU stream thread. |
| `tiny_duo_infer/backends/` | Backend protocol notes and MLX backend helpers; Phase 2 expansion target. |

## Tests And Scripts

| Path | Purpose |
|---|---|
| `tests/` | Unit, integration, CLI, serving, profiling, quantization (`tests/test_quantization.py`, `tests/test_quantization_integration.py`), and optional slow real-model smoke tests. |
| `scripts/` | Developer-facing benchmark and profiling entrypoints (`scripts/benchmark.py`, `scripts/profile_generation.py`). |

## Project Docs

| Path | Purpose |
|---|---|
| `docs/agent-guidelines.md` | Multi-agent collaboration rules, review gates, handoff format, and sign-off requirements. |
| `docs/architecture.md` | Source of truth for architecture details, subsystem responsibilities, and implementation boundaries. |
| `docs/file-structure.md` | This navigation map. |
| `docs/refined-plan.md` | Project strategy: purpose, roadmap status, durable decisions, and phase rationale. |
| `docs/phases/README.md` | Phase index: active phase pointer, completed phases, and deferred phases. |
| `docs/phases/` | Phase specs, taskboards, and handoffs, including active Phase 1.8 quantization docs. Completed phase docs are historical unless the active phase depends on them. |
| `docs/adr/` | Durable architecture decisions, if introduced. |

## Learning Materials

| Path | Purpose |
|---|---|
| `learning_materials/` | Human-oriented reading path and deep dives for learning the engine internals. |
| `learning_materials/deep_dives/` | Focused explanations of concepts such as RoPE, GQA/KV cache, sampling, and SwiGLU. |
