"""
Tests for `tiny_duo_infer.profiling`: argument parsing, prompt resolution,
context-policy forwarding, warmup handling, summary statistics, and the
JSON / human output shapes.

All tests use a `_FakeEngine` that returns deterministic GenerationStats so
the profiling logic can be exercised without loading any real model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pytest

from tiny_duo_infer.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationStats,
)
from tiny_duo_infer.profiling import (
    DEFAULT_PROMPTS,
    SCHEMA_VERSION,
    aggregate_runs,
    load_prompts,
    main,
    percentile,
    summarize_metric,
)


# ---------------------------------------------------------------------------
# Fake engine and helpers
# ---------------------------------------------------------------------------


@dataclass
class _StubConfig:
    """Minimal `ModelConfig` stand-in used by the fake engine.

    `tiny_duo_infer.profiling.run_profile` only reads the architectural
    dimensions and `model_type`, so the rest of the real `ModelConfig` is
    irrelevant for these unit tests.
    """

    model_type: str = "llama"
    n_layers: int = 16
    n_kv_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048


def _make_stats(
    *,
    prefill_ms: float = 12.0,
    time_to_first_token_ms: float = 15.0,
    decode_ms: float = 20.0,
    total_ms: float = 35.0,
    decode_tokens_per_sec: float = 100.0,
    kv_cache_active_bytes: int = 983_040,
    kv_cache_allocated_bytes: int = 67_108_864,
    prompt_tokens: int = 3,
    accepted_prompt_tokens: int = 3,
    generated_tokens: int = 4,
    context_policy: str = "allow_context_stop",
    stop_reason: str = "max_new_tokens",
    max_seq_len: int = 2048,
) -> GenerationStats:
    """Build a `GenerationStats` satisfying the dataclass invariants."""
    return GenerationStats(
        context_policy=context_policy,
        original_prompt_tokens=accepted_prompt_tokens,
        accepted_prompt_tokens=accepted_prompt_tokens,
        truncated_prompt_tokens=0,
        rejected_prompt_tokens=0,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        stop_reason=stop_reason,
        prompt_prepare_ms=0.5,
        prefill_ms=prefill_ms,
        time_to_first_token_ms=time_to_first_token_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        decode_tokens_per_sec=decode_tokens_per_sec,
        kv_cache_allocated_bytes=kv_cache_allocated_bytes,
        kv_cache_active_bytes=kv_cache_active_bytes,
        max_seq_len=max_seq_len,
        active_seq_len=accepted_prompt_tokens + generated_tokens,
        model_type="llama",
    )


@dataclass
class _FakeEngine:
    """Engine stand-in: records every `generate_request` call for assertions.

    By default each call returns `_make_stats()`. Tests can override the
    sequence by setting `_FakeEngine.next_stats_seq` to a list; values are
    consumed in order. Once exhausted, the default stats are returned.
    """

    config: _StubConfig = field(default_factory=_StubConfig)
    calls: list[GenerationRequest] = field(default_factory=list)

    next_stats_seq: list[GenerationStats] = field(default_factory=list)
    from_model_path_calls: list[tuple[Path, int]] = field(default_factory=list)

    @classmethod
    def make_factory(
        cls,
        *,
        next_stats_seq: list[GenerationStats] | None = None,
    ):
        """Return a class-like callable usable as `engine_cls=` in `main()`.

        The returned object supports `.from_model_path(path, max_seq_len=...)`
        like the real `Engine`, and stashes the constructed instance on the
        class so tests can inspect it after `main()` returns.
        """
        instances: list[_FakeEngine] = []

        class _Factory:
            instances_ref = instances
            from_model_path_calls: list[tuple[Path, int]] = []

            @classmethod
            def from_model_path(cls_, model_path, max_seq_len: int = 2048):
                cls_.from_model_path_calls.append((Path(model_path), max_seq_len))
                eng = _FakeEngine(
                    config=_StubConfig(max_seq_len=max_seq_len),
                    next_stats_seq=list(next_stats_seq or []),
                )
                instances.append(eng)
                return eng

        return _Factory

    def generate_request(self, request: GenerationRequest) -> GenerationResponse:
        self.calls.append(request)
        if self.next_stats_seq:
            stats = self.next_stats_seq.pop(0)
        else:
            stats = _make_stats()
        return GenerationResponse(
            text="hello",
            prompt_tokens=stats.prompt_tokens,
            generated_tokens=stats.generated_tokens,
            stop_reason=stats.stop_reason,
            stats=stats,
        )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def test_percentile_single_value():
    """A single-element list returns that value for any percentile."""
    assert percentile([42.0], 0) == 42.0
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0
    assert percentile([42.0], 100) == 42.0


def test_percentile_linear_interpolation_three_values():
    """Linear interpolation matches the standard convention used by numpy."""
    values = [1.0, 2.0, 3.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 2.0
    assert percentile(values, 100) == 3.0
    # rank = 0.95 * 2 = 1.9, between 2.0 and 3.0 with frac=0.9 → 2.9
    assert percentile(values, 95) == pytest.approx(2.9)


def test_percentile_empty_returns_zero():
    """An empty input returns 0.0 rather than raising."""
    assert percentile([], 50) == 0.0


def test_percentile_rejects_out_of_range_p():
    with pytest.raises(ValueError, match=r"in \[0, 100\]"):
        percentile([1.0, 2.0], 150.0)


def test_summarize_metric_returns_min_p50_p95_max_keys():
    """The summary dict contains exactly the spec-required keys."""
    summary = summarize_metric([10.0, 20.0, 30.0, 40.0, 50.0])
    assert set(summary.keys()) == {"min", "p50", "p95", "max"}
    assert summary["min"] == 10.0
    assert summary["max"] == 50.0


def test_aggregate_runs_covers_all_metric_keys():
    """`aggregate_runs` produces summary blocks for every tracked metric."""
    runs = [_make_stats() for _ in range(3)]
    summary = aggregate_runs(runs)
    expected_keys = {
        "prefill_ms",
        "time_to_first_token_ms",
        "decode_ms",
        "total_ms",
        "decode_tokens_per_sec",
        "kv_cache_active_bytes",
    }
    assert set(summary.keys()) == expected_keys
    for metric_summary in summary.values():
        assert set(metric_summary.keys()) == {"min", "p50", "p95", "max"}


# ---------------------------------------------------------------------------
# load_prompts
# ---------------------------------------------------------------------------


def test_load_prompts_uses_default_set_when_no_args(tmp_path):
    assert load_prompts(None, None) == list(DEFAULT_PROMPTS)
    assert load_prompts([], None) == list(DEFAULT_PROMPTS)


def test_load_prompts_uses_repeated_prompt_args():
    prompts = load_prompts(["one", "two", "three"], None)
    assert prompts == ["one", "two", "three"]


def test_load_prompts_reads_non_empty_lines_from_file(tmp_path):
    file = tmp_path / "prompts.txt"
    file.write_text("first prompt\n\n  \nsecond prompt\nthird prompt\n", encoding="utf-8")
    prompts = load_prompts(None, file)
    assert prompts == ["first prompt", "second prompt", "third prompt"]


def test_load_prompts_rejects_empty_file(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("\n  \n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no non-empty lines"):
        load_prompts(None, file)


def test_load_prompts_rejects_both_inputs(tmp_path):
    file = tmp_path / "p.txt"
    file.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_prompts(["one"], file)


# ---------------------------------------------------------------------------
# main(): argument parsing and validation
# ---------------------------------------------------------------------------


def test_main_requires_model_path():
    """argparse rejects missing --model-path with SystemExit."""
    with pytest.raises(SystemExit):
        main(["--runs", "1"], engine_cls=_FakeEngine.make_factory(), stdout=StringIO(), stderr=StringIO())


def test_main_rejects_runs_below_one():
    factory = _FakeEngine.make_factory()
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny", "--runs", "0"],
            engine_cls=factory,
            stdout=StringIO(),
            stderr=StringIO(),
        )


def test_main_rejects_negative_warmup():
    factory = _FakeEngine.make_factory()
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny", "--warmup-runs", "-1"],
            engine_cls=factory,
            stdout=StringIO(),
            stderr=StringIO(),
        )


def test_main_rejects_prompt_and_prompt_file_combination(tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("hello\n", encoding="utf-8")
    factory = _FakeEngine.make_factory()
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path", "models/tiny",
                "--prompt", "hi",
                "--prompt-file", str(pf),
            ],
            engine_cls=factory,
            stdout=StringIO(),
            stderr=StringIO(),
        )


def test_main_rejects_unknown_context_policy():
    factory = _FakeEngine.make_factory()
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path", "models/tiny",
                "--context-policy", "nope",
            ],
            engine_cls=factory,
            stdout=StringIO(),
            stderr=StringIO(),
        )
    # Argparse rejects before any model load.
    assert factory.from_model_path_calls == []


# ---------------------------------------------------------------------------
# main(): end-to-end behavior on the fake engine
# ---------------------------------------------------------------------------


def test_main_default_invocation_uses_default_prompt_set_and_one_warmup():
    """Defaults: 3 prompts × (1 warmup + 3 measured) = 12 generate_request calls."""
    factory = _FakeEngine.make_factory()
    stdout, stderr = StringIO(), StringIO()
    exit_code = main(
        ["--model-path", "models/tiny"],
        engine_cls=factory,
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    eng = factory.instances_ref[-1]
    expected_calls = len(DEFAULT_PROMPTS) * (1 + 3)
    assert len(eng.calls) == expected_calls
    # stdout has the human report, not JSON.
    assert "profile_generation" in stdout.getvalue()
    assert "metric" in stdout.getvalue()


def test_main_forwards_context_policy_to_every_request():
    factory = _FakeEngine.make_factory()
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--context-policy", "truncate_left",
            "--runs", "2",
            "--warmup-runs", "1",
        ],
        engine_cls=factory,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    eng = factory.instances_ref[-1]
    # All 3 calls (1 warmup + 2 measured) should carry the policy.
    assert len(eng.calls) == 3
    for req in eng.calls:
        assert req.context_policy == "truncate_left"


def test_main_forwards_sampling_args_to_every_request():
    factory = _FakeEngine.make_factory()
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--max-new-tokens", "16",
            "--temperature", "0.7",
            "--top-k", "10",
            "--top-p", "0.9",
            "--runs", "1",
            "--warmup-runs", "0",
        ],
        engine_cls=factory,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    eng = factory.instances_ref[-1]
    assert len(eng.calls) == 1
    req = eng.calls[0]
    assert req.max_new_tokens == 16
    assert req.temperature == pytest.approx(0.7)
    assert req.top_k == 10
    assert req.top_p == pytest.approx(0.9)


def test_main_warmup_runs_excluded_from_summary():
    """Per-prompt summary uses only the measured runs, not warmups."""
    # Pre-build a sequence of stats whose values clearly distinguish warmup
    # (high decode_ms) from measured (low decode_ms). If warmup leaks into
    # the summary, the min/p50/p95/max for decode_ms would all reflect 999.
    warmup = _make_stats(decode_ms=999.0)
    measured = [
        _make_stats(decode_ms=10.0),
        _make_stats(decode_ms=20.0),
        _make_stats(decode_ms=30.0),
    ]
    factory = _FakeEngine.make_factory(next_stats_seq=[warmup, *measured])
    stdout = StringIO()
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--runs", "3",
            "--warmup-runs", "1",
            "--json",
        ],
        engine_cls=factory,
        stdout=stdout,
        stderr=StringIO(),
    )
    payload = json.loads(stdout.getvalue())
    decode_summary = payload["prompts"][0]["summary"]["decode_ms"]
    assert decode_summary["min"] == 10.0
    assert decode_summary["max"] == 30.0
    # 999 must not appear anywhere in the measured-runs payload.
    runs_dump = json.dumps(payload["prompts"][0]["runs"])
    assert "999" not in runs_dump


def test_main_json_output_has_stable_schema():
    """`--json` produces a JSON document with the documented fields."""
    factory = _FakeEngine.make_factory()
    stdout = StringIO()
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--prompt", "there",
            "--runs", "2",
            "--warmup-runs", "0",
            "--json",
        ],
        engine_cls=factory,
        stdout=stdout,
        stderr=StringIO(),
    )
    payload = json.loads(stdout.getvalue())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["config"]["runs"] == 2
    assert payload["config"]["warmup_runs"] == 0
    assert payload["config"]["context_policy"] == "allow_context_stop"
    assert payload["engine_info"]["model_type"] == "llama"
    assert "kv_cache_allocated_bytes" in payload["engine_info"]
    assert len(payload["prompts"]) == 2
    for entry in payload["prompts"]:
        assert "prompt" in entry
        assert "runs" in entry
        assert len(entry["runs"]) == 2
        assert "summary" in entry
        for metric_key in (
            "prefill_ms",
            "time_to_first_token_ms",
            "decode_ms",
            "total_ms",
            "decode_tokens_per_sec",
            "kv_cache_active_bytes",
        ):
            assert metric_key in entry["summary"]
            assert set(entry["summary"][metric_key].keys()) == {
                "min", "p50", "p95", "max",
            }
    assert "overall_summary" in payload


def test_main_json_run_dicts_omit_decode_step_ms():
    """`decode_step_ms` is profiling detail that is left empty in T03; we omit it."""
    factory = _FakeEngine.make_factory()
    stdout = StringIO()
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--runs", "1",
            "--warmup-runs", "0",
            "--json",
        ],
        engine_cls=factory,
        stdout=stdout,
        stderr=StringIO(),
    )
    payload = json.loads(stdout.getvalue())
    run_dict = payload["prompts"][0]["runs"][0]
    assert "decode_step_ms" not in run_dict
    # Every other documented stats field must still be present.
    for key in (
        "prefill_ms",
        "time_to_first_token_ms",
        "decode_ms",
        "total_ms",
        "decode_tokens_per_sec",
        "kv_cache_active_bytes",
        "kv_cache_allocated_bytes",
        "context_policy",
        "original_prompt_tokens",
        "accepted_prompt_tokens",
        "truncated_prompt_tokens",
        "rejected_prompt_tokens",
        "prompt_tokens",
        "generated_tokens",
        "stop_reason",
        "max_seq_len",
        "active_seq_len",
    ):
        assert key in run_dict


def test_main_json_silences_progress_on_stdout():
    """Progress lines must not bleed into the JSON document on stdout."""
    factory = _FakeEngine.make_factory()
    stdout = StringIO()
    stderr = StringIO()
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--runs", "1",
            "--warmup-runs", "1",
            "--json",
        ],
        engine_cls=factory,
        stdout=stdout,
        stderr=stderr,
    )
    # stdout must parse as JSON.
    json.loads(stdout.getvalue())
    # Progress lines (if any) live on stderr; under --json they're silenced.
    assert "[profile]" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_main_human_mode_emits_progress_on_stderr():
    """Without --json, progress lines go to stderr, summary to stdout."""
    factory = _FakeEngine.make_factory()
    stdout = StringIO()
    stderr = StringIO()
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--runs", "2",
            "--warmup-runs", "1",
        ],
        engine_cls=factory,
        stdout=stdout,
        stderr=stderr,
    )
    err = stderr.getvalue()
    assert "[profile]" in err
    assert "warmup" in err
    out = stdout.getvalue()
    # Human report on stdout includes the summary table headers.
    assert "metric" in out
    assert "p50" in out
    assert "p95" in out
    assert "overall summary" in out


def test_main_loads_model_with_specified_max_seq_len():
    factory = _FakeEngine.make_factory()
    main(
        [
            "--model-path", "models/tiny",
            "--max-seq-len", "1024",
            "--prompt", "hi",
            "--runs", "1",
            "--warmup-runs", "0",
        ],
        engine_cls=factory,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert factory.from_model_path_calls == [(Path("models/tiny"), 1024)]


def test_main_engine_info_uses_canonical_kv_cache_formula():
    """`engine_info.kv_cache_allocated_bytes` matches the generation-module formula."""
    from tiny_duo_infer.generation import kv_cache_bytes

    factory = _FakeEngine.make_factory()
    stdout = StringIO()
    main(
        [
            "--model-path", "models/tiny",
            "--max-seq-len", "1024",
            "--prompt", "hi",
            "--runs", "1",
            "--warmup-runs", "0",
            "--json",
        ],
        engine_cls=factory,
        stdout=stdout,
        stderr=StringIO(),
    )
    payload = json.loads(stdout.getvalue())
    info = payload["engine_info"]
    expected = kv_cache_bytes(
        n_layers=info["n_layers"],
        n_kv_heads=info["n_kv_heads"],
        seq_len=info["max_seq_len"],
        head_dim=info["head_dim"],
        bytes_per_element=info["kv_cache_bytes_per_element"],
    )
    assert info["kv_cache_allocated_bytes"] == expected
