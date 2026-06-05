"""
Repeatable generation profiling for Phase 1.7 observability.

Loads one local model once and runs a fixed prompt set through
``Engine.generate_request()`` for ``runs`` measured iterations (plus optional
warmup runs). Reports min/p50/p95/max summaries for the latency, throughput,
and KV-cache memory fields exposed by ``GenerationStats``.

Phase 1.7 measures user-visible request timing at the engine boundary; this
module does not add new timers and does not import model internals. Every
metric it summarises comes from the engine's own ``GenerationStats``.

The accompanying script entrypoint lives at ``scripts/profile_generation.py``
and is a thin wrapper around :func:`main`.

Usage (default prompt set, three measured runs after one warmup):

    uv run python scripts/profile_generation.py \\
      --model-path ./models/llama-3.2-1b \\
      --runs 3 --warmup-runs 1

Usage (machine-readable for future comparison tests):

    uv run python scripts/profile_generation.py \\
      --model-path ./models/llama-3.2-1b \\
      --runs 5 --warmup-runs 1 --json > profile.json

Phase 1.7 explicitly does *not* gate on speedup thresholds. The point is
shape and stability of the measured numbers, not their absolute value, and
local Apple Silicon timing varies with warmup, thermal state, background
load, and lazy execution.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from tiny_duo_infer.engine import Engine
from tiny_duo_infer.generation import (
    GenerationRequest,
    GenerationStats,
    kv_cache_bytes as _kv_cache_bytes,  # re-exposed for benchmark.py reuse
)
from tiny_duo_infer.quantization import QuantizationConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stable JSON schema version. Bump when fields are added/renamed/removed so
# downstream comparison tooling can branch on it.
# v2: Phase 1.8 adds quantization_mode, quantization_bits, quantization_group_size,
#     linear_weight_full_precision_bytes, and linear_weight_runtime_bytes to engine_info.
SCHEMA_VERSION: int = 2

# Built-in prompt set when neither --prompt nor --prompt-file is provided.
# Three short prompts that exercise different prompt lengths and stop reasons
# without requiring any external file. Kept generic enough to work across
# Llama and Qwen3 family models.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "Hello, world!",
    "Explain why the sky is blue.",
    "List three benefits of regular exercise:",
)

# Order of metrics summarised in both human and JSON output. Stable ordering
# matters for the human table and for downstream diffing of JSON dumps.
_METRIC_KEYS: tuple[str, ...] = (
    "prefill_ms",
    "time_to_first_token_ms",
    "decode_ms",
    "total_ms",
    "decode_tokens_per_sec",
    "kv_cache_active_bytes",
)

# Choices for --context-policy, kept in lockstep with the CLI and the
# `ContextPolicy` literal in `tiny_duo_infer.generation`.
_CONTEXT_POLICY_CHOICES: tuple[str, ...] = (
    "allow_context_stop",
    "reject",
    "truncate_left",
    "truncate_right",
    "reserve_generation",
)

# Quantization choices, mirroring tiny_duo_infer.cli._QUANTIZATION_CHOICES.
_QUANTIZATION_CHOICES: tuple[str, ...] = ("none", "int4", "int8")


def _positive_int(value: str) -> int:
    """Argparse type that rejects zero and negative group sizes before model load."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_quantization_config(args: argparse.Namespace) -> QuantizationConfig | None:
    """Translate --quantization / --quant-group-size into a QuantizationConfig."""
    if args.quantization == "none":
        return None
    bits = 4 if args.quantization == "int4" else 8
    return QuantizationConfig(bits=bits, group_size=args.quant_group_size)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    """Return the linear-interpolated percentile of ``values``.

    Uses the same convention as numpy's default (``method="linear"``):
    for a sorted sequence ``s`` of length ``n``, the rank is
    ``r = p/100 * (n - 1)``, and the percentile is
    ``s[floor(r)] + frac(r) * (s[ceil(r)] - s[floor(r)])``.

    Edge cases:
      - empty input returns 0.0 (callers should not feed empty lists; this
        keeps the JSON output well-formed if a metric is unexpectedly missing).
      - single value returns that value for any percentile.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p must be in [0, 100], got {p!r}.")

    s = sorted(float(v) for v in values)
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def summarize_metric(values: list[float]) -> dict[str, float]:
    """Return ``{"min": ..., "p50": ..., "p95": ..., "max": ...}`` for ``values``.

    Spec line 438: "summaries report min, p50, p95, and max". The keys
    here are the spec field names; downstream consumers can rely on them.
    """
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": float(max(values)),
    }


def aggregate_runs(runs: list[GenerationStats]) -> dict[str, dict[str, float]]:
    """Aggregate per-metric summaries across a list of `GenerationStats`.

    Returns a dict keyed by metric name (in `_METRIC_KEYS` order) → summary.
    Memory-byte metrics are still aggregated as floats; downstream consumers
    can cast back to int if needed. Keeping a uniform value type simplifies
    the JSON shape.
    """
    result: dict[str, dict[str, float]] = {}
    for key in _METRIC_KEYS:
        result[key] = summarize_metric([float(getattr(r, key)) for r in runs])
    return result


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(
    prompt_args: list[str] | None,
    prompt_file: Path | None,
) -> list[str]:
    """Resolve the active prompt set from CLI args.

    Priority:
      1. ``--prompt`` (one or more) — used as-is in the order given.
      2. ``--prompt-file`` — one prompt per non-empty line, trailing
         whitespace stripped per line; blank/whitespace-only lines skipped.
      3. Built-in :data:`DEFAULT_PROMPTS` set.

    ``--prompt`` and ``--prompt-file`` are mutually exclusive at the
    argparse layer; this function still checks defensively.
    """
    if prompt_args and prompt_file is not None:
        # Defensive: argparse should have caught this via parser.error().
        raise ValueError("--prompt and --prompt-file are mutually exclusive.")

    if prompt_args:
        return list(prompt_args)

    if prompt_file is not None:
        text = prompt_file.read_text(encoding="utf-8")
        prompts = [line.rstrip() for line in text.splitlines()]
        prompts = [p for p in prompts if p.strip()]
        if not prompts:
            raise ValueError(
                f"prompt file {prompt_file} contains no non-empty lines."
            )
        return prompts

    return list(DEFAULT_PROMPTS)


# ---------------------------------------------------------------------------
# Profile run
# ---------------------------------------------------------------------------


def _build_request(prompt: str, args: argparse.Namespace) -> GenerationRequest:
    """Construct a `GenerationRequest` from parsed CLI args + a single prompt.

    Validation (e.g. unknown context_policy, out-of-range temperature) is
    delegated to `GenerationRequest.__post_init__`, which mirrors the CLI
    behavior. Errors propagate so the script exits with a clear message.
    """
    return GenerationRequest(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        context_policy=args.context_policy,
    )


def run_profile(
    engine: Engine,
    prompts: list[str],
    *,
    args: argparse.Namespace,
    progress: TextIO | None = None,
) -> dict:
    """Drive the profile loop and return a structured result dict.

    For each prompt, runs ``warmup_runs`` warmup iterations (discarded) and
    ``runs`` measured iterations, then aggregates per-prompt summaries plus
    one overall summary across every measured run.

    The ``progress`` stream, when provided, receives one line per measured
    run and one summary line per prompt. The default human renderer wires
    this to stderr so ``--json`` output on stdout stays clean.
    """
    if args.runs < 1:
        raise ValueError(f"--runs must be >= 1, got {args.runs}.")
    if args.warmup_runs < 0:
        raise ValueError(
            f"--warmup-runs must be >= 0, got {args.warmup_runs}."
        )

    cfg = engine.config
    # KV-cache allocation is determined by max_seq_len + model dims and is
    # constant across runs; report it once so the JSON consumer doesn't need
    # to dedupe it across every measured run.
    bytes_per_element = _bytes_per_element(engine)
    quant = getattr(engine, "_quantization", None)
    ws = getattr(engine, "_linear_weight_stats", None)
    quant_mode = "none" if quant is None else f"int{quant.bits}"
    engine_info = {
        "model_type": cfg.model_type,
        "max_seq_len": cfg.max_seq_len,
        "n_layers": cfg.n_layers,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "kv_cache_allocated_bytes": _kv_cache_bytes(
            n_layers=cfg.n_layers,
            n_kv_heads=cfg.n_kv_heads,
            seq_len=cfg.max_seq_len,
            head_dim=cfg.head_dim,
            bytes_per_element=bytes_per_element,
        ),
        "kv_cache_bytes_per_element": bytes_per_element,
        "quantization_mode": quant_mode,
        "quantization_bits": quant.bits if quant is not None else None,
        "quantization_group_size": quant.group_size if quant is not None else None,
        "linear_weight_full_precision_bytes": ws.linear_weight_full_precision_bytes if ws else 0,
        "linear_weight_runtime_bytes": ws.linear_weight_runtime_bytes if ws else 0,
    }

    per_prompt: list[dict] = []
    all_measured: list[GenerationStats] = []

    for prompt_index, prompt in enumerate(prompts):
        request = _build_request(prompt, args)

        if progress is not None:
            print(
                f"[profile] prompt {prompt_index + 1}/{len(prompts)}:"
                f" {prompt!r}",
                file=progress,
            )

        # Warmup: results discarded. We still run the same request so the
        # MLX graph and KV buffers are warm by the time measurement starts.
        for w in range(args.warmup_runs):
            engine.generate_request(request)
            if progress is not None:
                print(f"[profile]   warmup {w + 1}/{args.warmup_runs}", file=progress)

        runs: list[GenerationStats] = []
        for r in range(args.runs):
            response = engine.generate_request(request)
            stats = response.stats
            if stats is None:
                raise RuntimeError(
                    "engine returned a response without GenerationStats; "
                    "Phase 1.7-T03 requires every real engine path to "
                    "populate stats."
                )
            runs.append(stats)
            all_measured.append(stats)
            if progress is not None:
                print(
                    f"[profile]   run {r + 1}/{args.runs}"
                    f"  prefill={stats.prefill_ms:.2f}ms"
                    f"  ttft={stats.time_to_first_token_ms:.2f}ms"
                    f"  decode={stats.decode_ms:.2f}ms"
                    f"  total={stats.total_ms:.2f}ms"
                    f"  tok/s={stats.decode_tokens_per_sec:.2f}",
                    file=progress,
                )

        per_prompt.append(
            {
                "prompt": prompt,
                "runs": [_stats_to_dict(s) for s in runs],
                "summary": aggregate_runs(runs),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "engine_info": engine_info,
        "config": {
            "model_path": str(args.model_path),
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "context_policy": args.context_policy,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "quantization": args.quantization,
            "quant_group_size": args.quant_group_size,
        },
        "prompts": per_prompt,
        "overall_summary": aggregate_runs(all_measured),
    }


def _bytes_per_element(engine: Engine) -> int:
    """Best-effort bytes-per-KV-element using the live cache when available.

    T03 already computes ``kv_cache_active_bytes`` from the live cache dtype,
    so per-run bytes are correct regardless of dtype. This helper is only
    used for the engine-level *allocated* total, which depends on the same
    dtype. We fall back to the canonical fp32 default if the engine has no
    cache attached (which only happens in tests with stub engines).
    """
    cache = getattr(engine, "_cache", None) or getattr(engine, "cache", None)
    keys = getattr(cache, "_keys", None) if cache is not None else None
    if keys:
        try:
            return int(keys[0].dtype.size)
        except AttributeError:
            pass
    # Match `tiny_duo_infer.generation.kv_cache_bytes` default.
    return 4


def _stats_to_dict(stats: GenerationStats) -> dict:
    """Serialize a `GenerationStats` to a JSON-safe dict.

    Drops `decode_step_ms` because Phase 1.7 leaves it empty in non-profiling
    paths (T03 note); leaving it in the JSON would just emit empty arrays
    everywhere.
    """
    raw = asdict(stats)
    raw.pop("decode_step_ms", None)
    return raw


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def format_json(result: dict) -> str:
    """Render the profile result as a stable, indented JSON string."""
    return json.dumps(result, indent=2, sort_keys=False)


def format_human(result: dict) -> str:
    """Render the profile result as a human-readable text block."""
    lines: list[str] = []
    cfg = result["config"]
    info = result["engine_info"]

    lines.append("=== profile_generation ===")
    lines.append(f"model_path     : {cfg['model_path']}")
    lines.append(f"model_type     : {info['model_type']}")
    lines.append(
        f"max_seq_len    : {cfg['max_seq_len']}"
        f"  (kv_cache_allocated_bytes={info['kv_cache_allocated_bytes']:,})"
    )
    lines.append(
        f"sampling       : max_new_tokens={cfg['max_new_tokens']}"
        f"  temperature={cfg['temperature']}"
        f"  top_k={cfg['top_k']}  top_p={cfg['top_p']}"
    )
    lines.append(f"context_policy : {cfg['context_policy']}")
    lines.append(
        f"runs           : {cfg['runs']} measured"
        f" + {cfg['warmup_runs']} warmup"
    )

    # Quantization and weight-memory summary (Phase 1.8)
    quant_mode = info.get("quantization_mode", "none")
    fp_bytes = info.get("linear_weight_full_precision_bytes", 0)
    rt_bytes = info.get("linear_weight_runtime_bytes", 0)
    if quant_mode == "none":
        lines.append(
            f"quantization   : none"
            f"  (linear_weight_bytes={fp_bytes:,})"
        )
    else:
        quant_gs = info.get("quantization_group_size", "?")
        if fp_bytes > 0:
            pct = (rt_bytes - fp_bytes) / fp_bytes * 100.0
            mem_str = (
                f"{rt_bytes:,} bytes runtime vs {fp_bytes:,} full-precision"
                f" ({pct:+.1f}%)"
            )
        else:
            mem_str = f"{rt_bytes:,} bytes runtime"
        lines.append(
            f"quantization   : {quant_mode} group_size={quant_gs}"
            f"  weight_memory={mem_str}"
        )
    lines.append("")

    for entry in result["prompts"]:
        lines.append(f"--- prompt: {entry['prompt']!r} ---")
        lines.extend(_format_summary_table(entry["summary"]))
        lines.append("")

    lines.append("--- overall summary ---")
    lines.extend(_format_summary_table(result["overall_summary"]))
    return "\n".join(lines)


def _format_summary_table(summary: dict[str, dict[str, float]]) -> list[str]:
    """Format one summary block as a left-aligned table of fixed columns."""
    header = f"  {'metric':<28}{'min':>14}{'p50':>14}{'p95':>14}{'max':>14}"
    sep = "  " + "-" * (28 + 14 * 4)
    rows = [header, sep]
    for key in _METRIC_KEYS:
        s = summary.get(key, {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0})
        rows.append(
            f"  {key:<28}"
            f"{_fmt_value(key, s['min']):>14}"
            f"{_fmt_value(key, s['p50']):>14}"
            f"{_fmt_value(key, s['p95']):>14}"
            f"{_fmt_value(key, s['max']):>14}"
        )
    return rows


def _fmt_value(key: str, value: float) -> str:
    """Format one cell: bytes as integers, latency/throughput as 2-decimal floats."""
    if key.endswith("_bytes"):
        return f"{int(round(value)):,}"
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    engine_cls: type[Engine] = Engine,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Parse args, load the model once, run the profile, and render output.

    Args:
        argv:       command-line arguments excluding the program name.
        engine_cls: Engine-compatible class, injectable for tests.
        stdout:     receives the final report (text by default, JSON when
                    ``--json`` is set).
        stderr:     receives progress lines and warmup summaries so stdout
                    stays a clean, parseable stream.

    Returns:
        Process exit code. ``0`` on success.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.prompt and args.prompt_file is not None:
        parser.error("--prompt and --prompt-file are mutually exclusive")
    if args.runs < 1:
        parser.error(f"--runs must be >= 1, got {args.runs}")
    if args.warmup_runs < 0:
        parser.error(f"--warmup-runs must be >= 0, got {args.warmup_runs}")

    try:
        prompts = load_prompts(args.prompt or None, args.prompt_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    engine = engine_cls.from_model_path(
        Path(args.model_path),
        max_seq_len=args.max_seq_len,
        quantization=_build_quantization_config(args),
    )

    progress = None if args.json else stderr
    try:
        result = run_profile(engine, prompts, args=args, progress=progress)
    except ValueError as exc:
        # Surface request-construction failures (e.g. bad context policy)
        # as a clean error rather than a stack trace.
        parser.error(str(exc))

    if args.json:
        print(format_json(result), file=stdout)
    else:
        print(format_human(result), file=stdout)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the profiling CLI."""
    parser = argparse.ArgumentParser(
        prog="profile_generation",
        description=(
            "Repeatable generation profiling: load one model, run a fixed "
            "prompt set across N measured iterations (plus optional warmup), "
            "and report min/p50/p95/max summaries for latency, throughput, "
            "and KV-cache memory."
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a local HuggingFace-compatible model directory.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum total sequence length for the KV cache. Default: 2048.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate per run. Default: 64.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature. 0.0 selects greedy decoding for "
            "deterministic profiling. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k cutoff. 0 disables. Default: 0.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p cutoff. 1.0 disables. Default: 1.0.",
    )
    parser.add_argument(
        "--context-policy",
        choices=_CONTEXT_POLICY_CHOICES,
        default="allow_context_stop",
        help=(
            "Per-request context-budget policy. Default: allow_context_stop."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per prompt, after warmup. Default: 3.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help=(
            "Warmup runs per prompt, executed before measurement and "
            "excluded from the summary. Default: 1."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help=(
            "Prompt text. May be repeated. If neither --prompt nor "
            "--prompt-file is set, a built-in default prompt set is used."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "Path to a text file with one prompt per non-empty line. "
            "Mutually exclusive with --prompt."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit machine-readable JSON to stdout instead of the human "
            "report. Progress lines (warmup, per-run) are suppressed when "
            "--json is set so stdout stays a clean JSON document."
        ),
    )
    parser.add_argument(
        "--quantization",
        choices=_QUANTIZATION_CHOICES,
        default="none",
        help=(
            "Weight-only quantization mode used when loading the model. "
            "none = full precision (default); int4 = INT4; int8 = INT8. "
            "Use this flag to compare full-precision and quantized profiles."
        ),
    )
    parser.add_argument(
        "--quant-group-size",
        type=_positive_int,
        default=64,
        help=(
            "Quantization group size. Must evenly divide every quantized "
            "weight's input dimension. Default: 64."
        ),
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
