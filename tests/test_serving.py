"""
Tests for tiny_duo_infer.serving.api.

The server is tested via FastAPI's TestClient (httpx-backed sync client).
All unit tests use _FakeEngine so no model artifacts are required.

Streaming format: one NDJSON line per event.
  Fragment: {"done": false, "text": "<fragment>"}
  Final:    {"done": true, "text": "<full trimmed text>", "prompt_tokens": N,
             "generated_tokens": N, "stop_reason": "<reason>", "stats": {…}|null}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tiny_duo_infer.serving.api as api_module
import tiny_duo_infer.serving.worker as worker_module
from tiny_duo_infer.engine import Engine
from tiny_duo_infer.generation import GenerationRequest, GenerationResponse, GenerationStats
from tiny_duo_infer.quantization import QuantizationConfig
from tiny_duo_infer.serving.api import create_app, create_app_from_path
from tiny_duo_infer.serving.worker import InferenceWorker


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Fake engine that returns canned text for all requests."""

    def generate_request(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="hello world",
            prompt_tokens=3,
            generated_tokens=2,
            stop_reason="eos",
        )

    def generate_stream(self, request: GenerationRequest):
        yield "hel"
        yield "lo"
        yield " world"
        yield GenerationResponse(
            text="hello world",
            prompt_tokens=3,
            generated_tokens=3,
            stop_reason="eos",
        )


class _SlowFakeEngine:
    """Fake engine that blocks until released, used for busy-response tests."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_request(self, request: GenerationRequest) -> GenerationResponse:
        self.started.set()
        self.release.wait()
        return GenerationResponse(
            text="done",
            prompt_tokens=1,
            generated_tokens=1,
            stop_reason="eos",
        )

    def generate_stream(self, request: GenerationRequest):
        self.started.set()
        self.release.wait()
        yield "done"
        yield GenerationResponse(
            text="done",
            prompt_tokens=1,
            generated_tokens=1,
            stop_reason="eos",
        )


@pytest.fixture(autouse=True)
def reset_server_state():
    """Shut down any worker created in a previous test and clear module state."""
    old_worker = api_module._worker
    api_module._worker = None
    yield
    if api_module._worker is not None:
        api_module._worker.shutdown()
    api_module._worker = old_worker


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_worker_not_initialized_returns_500_cleanly():
    """Requests before create_app() fail cleanly with 500, not 503."""
    from tiny_duo_infer.serving.api import app as _app

    api_module._worker = None
    client = TestClient(_app, raise_server_exceptions=False)

    # First request: worker not initialised → 500
    resp1 = client.post("/generate", json={"prompt": "hi"})
    assert resp1.status_code == 500

    # Second request: also returns 500, not 503 (no busy state leaked)
    resp2 = client.post("/generate", json={"prompt": "hi"})
    assert resp2.status_code != 503


def test_health_returns_ok_when_idle():
    """GET /health reports status=ok and active=False when no request is running."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["active"] is False


def test_health_shows_active_while_request_running():
    """GET /health reports active=True while a generation request is active."""
    slow = _SlowFakeEngine()
    client = TestClient(create_app(slow), raise_server_exceptions=False)
    results: dict[str, object] = {}

    def make_request():
        results["gen"] = client.post("/generate", json={"prompt": "hi"})

    t = threading.Thread(target=make_request)
    t.start()
    slow.started.wait(timeout=5)

    resp = client.get("/health")
    assert resp.json()["active"] is True

    slow.release.set()
    t.join(timeout=5)


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------


def test_generate_returns_text_and_metadata():
    """POST /generate returns the full text and response metadata as JSON."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate", json={"prompt": "hi", "temperature": 0.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hello world"
    assert body["prompt_tokens"] == 3
    assert body["generated_tokens"] == 2
    assert body["stop_reason"] == "eos"


def test_generate_with_messages():
    """POST /generate with messages field wires chat=True through to the engine."""
    received: list[GenerationRequest] = []

    class _RecordingEngine:
        def generate_request(self, request: GenerationRequest) -> GenerationResponse:
            received.append(request)
            return GenerationResponse(
                text="ok", prompt_tokens=1, generated_tokens=1, stop_reason="eos"
            )

    client = TestClient(create_app(_RecordingEngine()))
    resp = client.post(
        "/generate",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "chat": True,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    assert len(received) == 1
    req = received[0]
    assert req.chat is True
    assert req.messages is not None
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "Hi"


def test_generate_invalid_request_returns_422():
    """POST /generate with top_p=0.0 fails GenerationRequest validation → 422."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post(
        "/generate",
        json={"prompt": "hi", "top_p": 0.0},
    )
    assert resp.status_code == 422


def test_generate_returns_503_when_busy():
    """POST /generate returns 503 while another request is active."""
    slow = _SlowFakeEngine()
    client = TestClient(create_app(slow), raise_server_exceptions=False)
    results: dict[str, object] = {}

    def first_request():
        results["first"] = client.post("/generate", json={"prompt": "hi"})

    t = threading.Thread(target=first_request)
    t.start()
    slow.started.wait(timeout=5)

    results["second"] = client.post("/generate", json={"prompt": "hi"})

    slow.release.set()
    t.join(timeout=5)

    assert results["second"].status_code == 503
    assert "busy" in results["second"].json()["detail"]
    assert results["first"].status_code == 200


# ---------------------------------------------------------------------------
# POST /generate/stream
# ---------------------------------------------------------------------------


def _parse_ndjson(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_generate_stream_yields_fragments_in_order():
    """POST /generate/stream sends one NDJSON line per fragment in generation order."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate/stream", json={"prompt": "hi"})
    assert resp.status_code == 200
    chunks = _parse_ndjson(resp.text)
    fragment_texts = [c["text"] for c in chunks if not c["done"]]
    assert fragment_texts == ["hel", "lo", " world"]


def test_generate_stream_final_chunk_has_metadata():
    """The last NDJSON chunk has done=True, full trimmed text, and token metadata."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate/stream", json={"prompt": "hi"})
    chunks = _parse_ndjson(resp.text)
    final = chunks[-1]
    assert final["done"] is True
    assert final["text"] == "hello world"
    assert final["prompt_tokens"] == 3
    assert final["generated_tokens"] == 3
    assert final["stop_reason"] == "eos"


def test_generate_stream_fragment_chunks_have_done_false():
    """All non-final NDJSON chunks have done=False."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate/stream", json={"prompt": "hi"})
    chunks = _parse_ndjson(resp.text)
    for chunk in chunks[:-1]:
        assert chunk["done"] is False


def test_generate_stream_returns_503_when_busy():
    """POST /generate/stream returns 503 while another request is active."""
    slow = _SlowFakeEngine()
    client = TestClient(create_app(slow), raise_server_exceptions=False)
    results: dict[str, object] = {}

    def first_request():
        results["first"] = client.post("/generate/stream", json={"prompt": "hi"})

    t = threading.Thread(target=first_request)
    t.start()
    slow.started.wait(timeout=5)

    results["second"] = client.post("/generate/stream", json={"prompt": "hi"})

    slow.release.set()
    t.join(timeout=5)

    assert results["second"].status_code == 503
    assert "busy" in results["second"].json()["detail"]


def test_generate_stream_invalid_request_returns_422():
    """POST /generate/stream with invalid request fields returns 422."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate/stream", json={"prompt": "hi", "top_p": 0.0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Slow smoke tests (require local model artifacts)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_qwen3_generate_smoke():
    """POST /generate against real Qwen3 model completes and returns metadata.

    Uses create_app_from_path so the engine is initialised inside the
    InferenceWorker thread — MLX GPU stream stays on that thread throughout.
    """
    import os
    from pathlib import Path

    model_path = Path(os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b"))
    client = TestClient(create_app_from_path(model_path, max_seq_len=512))

    resp = client.post(
        "/generate",
        json={"prompt": "The capital of France is", "max_new_tokens": 2, "temperature": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["text"], str)
    assert body["stop_reason"] in ("eos", "max_new_tokens", "context_length")
    assert body["stats"] is not None
    assert body["stats"]["prompt_prepare_ms"] >= 0
    assert body["stats"]["prefill_ms"] >= 0
    assert body["stats"]["total_ms"] >= 0
    assert "decode_step_ms" not in body["stats"]


@pytest.mark.slow
def test_qwen3_generate_stream_smoke():
    """POST /generate/stream against real Qwen3 model returns valid NDJSON.

    Uses create_app_from_path so the engine is initialised inside the
    InferenceWorker thread — MLX GPU stream stays on that thread throughout.
    """
    import os
    from pathlib import Path

    model_path = Path(os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b"))
    client = TestClient(create_app_from_path(model_path, max_seq_len=512))

    resp = client.post(
        "/generate/stream",
        json={"prompt": "Hello", "max_new_tokens": 2, "temperature": 0.0},
    )
    assert resp.status_code == 200
    chunks = _parse_ndjson(resp.text)
    assert chunks[-1]["done"] is True
    assert chunks[-1]["stop_reason"] in ("eos", "max_new_tokens", "context_length")
    assert chunks[-1]["stats"] is not None
    assert "decode_step_ms" not in chunks[-1]["stats"]


# ---------------------------------------------------------------------------
# Test doubles for T05 stats and context_policy tests
# ---------------------------------------------------------------------------


def _make_test_stats() -> GenerationStats:
    return GenerationStats(
        context_policy="allow_context_stop",
        original_prompt_tokens=3,
        accepted_prompt_tokens=3,
        truncated_prompt_tokens=0,
        rejected_prompt_tokens=0,
        prompt_tokens=3,
        generated_tokens=2,
        stop_reason="eos",
        prompt_prepare_ms=1.0,
        prefill_ms=10.0,
        time_to_first_token_ms=15.0,
        decode_ms=20.0,
        total_ms=35.0,
        decode_tokens_per_sec=100.0,
        kv_cache_allocated_bytes=8192,
        kv_cache_active_bytes=2048,
        max_seq_len=100,
        active_seq_len=5,
        model_type="llama",
    )


class _FakeEngineWithStats:
    """Fake engine that returns GenerationResponse with a populated stats object."""

    def generate_request(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="hello world",
            prompt_tokens=3,
            generated_tokens=2,
            stop_reason="eos",
            stats=_make_test_stats(),
        )

    def generate_stream(self, request: GenerationRequest):
        yield "hello"
        yield " world"
        yield GenerationResponse(
            text="hello world",
            prompt_tokens=3,
            generated_tokens=2,
            stop_reason="eos",
            stats=_make_test_stats(),
        )


# ---------------------------------------------------------------------------
# POST /generate — stats field
# ---------------------------------------------------------------------------


def test_generate_includes_stats_when_engine_provides_them():
    """POST /generate includes stats object when engine populates GenerationResponse.stats."""
    client = TestClient(create_app(_FakeEngineWithStats()))
    resp = client.post("/generate", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["stats"] is not None


def test_generate_stats_is_null_when_engine_returns_none():
    """POST /generate returns stats=null when engine does not populate stats (e.g. fake engines)."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["stats"] is None


def test_generate_stats_has_required_fields():
    """POST /generate stats object contains all required fields."""
    client = TestClient(create_app(_FakeEngineWithStats()))
    resp = client.post("/generate", json={"prompt": "hi"})
    stats = resp.json()["stats"]
    required_fields = [
        "context_policy",
        "original_prompt_tokens",
        "accepted_prompt_tokens",
        "truncated_prompt_tokens",
        "rejected_prompt_tokens",
        "prompt_tokens",
        "generated_tokens",
        "stop_reason",
        "prompt_prepare_ms",
        "prefill_ms",
        "time_to_first_token_ms",
        "decode_ms",
        "total_ms",
        "decode_tokens_per_sec",
        "kv_cache_allocated_bytes",
        "kv_cache_active_bytes",
        "max_seq_len",
        "active_seq_len",
        "model_type",
    ]
    for f in required_fields:
        assert f in stats, f"missing required stats field: {f!r}"


def test_generate_stats_omits_decode_step_ms():
    """POST /generate stats must not include decode_step_ms (profiling-only field)."""
    client = TestClient(create_app(_FakeEngineWithStats()))
    resp = client.post("/generate", json={"prompt": "hi"})
    assert "decode_step_ms" not in resp.json()["stats"]


# ---------------------------------------------------------------------------
# POST /generate — context_policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    ["allow_context_stop", "reject", "truncate_left", "truncate_right", "reserve_generation"],
)
def test_generate_accepts_all_valid_context_policies(policy):
    """POST /generate accepts every valid context_policy value and returns 200."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate", json={"prompt": "hi", "context_policy": policy})
    assert resp.status_code == 200


def test_generate_forwards_context_policy_to_engine():
    """context_policy from the HTTP request body is forwarded to GenerationRequest."""
    received: list[GenerationRequest] = []

    class _RecordingEngine:
        def generate_request(self, request: GenerationRequest) -> GenerationResponse:
            received.append(request)
            return GenerationResponse(
                text="ok", prompt_tokens=1, generated_tokens=1, stop_reason="eos"
            )

    client = TestClient(create_app(_RecordingEngine()))
    client.post("/generate", json={"prompt": "hi", "context_policy": "reject"})
    assert len(received) == 1
    assert received[0].context_policy == "reject"


def test_generate_invalid_context_policy_returns_422():
    """POST /generate with an unknown context_policy returns 422."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate", json={"prompt": "hi", "context_policy": "not_a_policy"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /generate/stream — stats field
# ---------------------------------------------------------------------------


def test_generate_stream_final_chunk_includes_stats():
    """The final NDJSON chunk includes a stats object when the engine provides one."""
    client = TestClient(create_app(_FakeEngineWithStats()))
    resp = client.post("/generate/stream", json={"prompt": "hi"})
    chunks = _parse_ndjson(resp.text)
    final = chunks[-1]
    assert final["done"] is True
    assert final["stats"] is not None
    assert "prompt_prepare_ms" in final["stats"]
    assert "prefill_ms" in final["stats"]
    assert "time_to_first_token_ms" in final["stats"]


def test_generate_stream_final_chunk_stats_null_when_engine_returns_none():
    """The final NDJSON chunk has stats=null when the engine does not populate stats."""
    client = TestClient(create_app(_FakeEngine()))
    resp = client.post("/generate/stream", json={"prompt": "hi"})
    chunks = _parse_ndjson(resp.text)
    assert chunks[-1]["stats"] is None


def test_generate_stream_fragment_chunks_omit_stats():
    """Fragment NDJSON chunks (done=false) must not include a stats key."""
    client = TestClient(create_app(_FakeEngineWithStats()))
    resp = client.post("/generate/stream", json={"prompt": "hi"})
    chunks = _parse_ndjson(resp.text)
    for chunk in chunks[:-1]:
        assert chunk["done"] is False
        assert "stats" not in chunk


def test_generate_stream_final_chunk_stats_omits_decode_step_ms():
    """Final NDJSON chunk stats must not include decode_step_ms."""
    client = TestClient(create_app(_FakeEngineWithStats()))
    resp = client.post("/generate/stream", json={"prompt": "hi"})
    chunks = _parse_ndjson(resp.text)
    assert "decode_step_ms" not in chunks[-1]["stats"]


# ---------------------------------------------------------------------------
# Quantization forwarding — T05
# ---------------------------------------------------------------------------


def test_worker_from_path_forwards_quantization_none_to_engine(monkeypatch):
    """InferenceWorker.from_path with quantization=None passes None to Engine."""
    records: list[dict] = []

    def fake_from_model_path(model_path, max_seq_len=2048, quantization=None):
        records.append({"model_path": model_path, "max_seq_len": max_seq_len, "quantization": quantization})
        return _FakeEngine()

    monkeypatch.setattr(Engine, "from_model_path", fake_from_model_path)

    worker = InferenceWorker.from_path("/fake/model", max_seq_len=64, quantization=None)
    worker.shutdown()

    assert len(records) == 1
    assert records[0]["quantization"] is None


def test_worker_from_path_forwards_quantization_config_to_engine(monkeypatch):
    """InferenceWorker.from_path passes QuantizationConfig to Engine inside the worker thread."""
    records: list[dict] = []
    quant_config = QuantizationConfig(bits=4, group_size=64)

    def fake_from_model_path(model_path, max_seq_len=2048, quantization=None):
        records.append({"model_path": model_path, "max_seq_len": max_seq_len, "quantization": quantization})
        return _FakeEngine()

    monkeypatch.setattr(Engine, "from_model_path", fake_from_model_path)

    worker = InferenceWorker.from_path("/fake/model", max_seq_len=64, quantization=quant_config)
    worker.shutdown()

    assert len(records) == 1
    assert records[0]["quantization"] is quant_config
    assert records[0]["quantization"].bits == 4
    assert records[0]["quantization"].group_size == 64


def test_worker_from_path_forwards_int8_quantization_config(monkeypatch):
    """InferenceWorker.from_path forwards INT8 QuantizationConfig correctly."""
    records: list[dict] = []
    quant_config = QuantizationConfig(bits=8, group_size=32)

    def fake_from_model_path(model_path, max_seq_len=2048, quantization=None):
        records.append({"quantization": quantization})
        return _FakeEngine()

    monkeypatch.setattr(Engine, "from_model_path", fake_from_model_path)

    worker = InferenceWorker.from_path("/fake/model", max_seq_len=64, quantization=quant_config)
    worker.shutdown()

    assert records[0]["quantization"].bits == 8
    assert records[0]["quantization"].group_size == 32


def test_create_app_from_path_forwards_quantization_to_worker(monkeypatch):
    """create_app_from_path passes quantization through to InferenceWorker.from_path."""
    worker_calls: list[dict] = []
    quant_config = QuantizationConfig(bits=4, group_size=64)

    original_from_path = InferenceWorker.from_path

    def fake_from_path(cls_or_path, max_seq_len=2048, quantization=None, **kwargs):
        worker_calls.append({"max_seq_len": max_seq_len, "quantization": quantization})
        return InferenceWorker.from_engine(_FakeEngine())

    monkeypatch.setattr(InferenceWorker, "from_path", fake_from_path)

    create_app_from_path(Path("/fake/model"), max_seq_len=512, quantization=quant_config)

    assert len(worker_calls) == 1
    assert worker_calls[0]["quantization"] is quant_config
    assert worker_calls[0]["max_seq_len"] == 512


def test_create_app_from_path_default_quantization_is_none(monkeypatch):
    """create_app_from_path defaults to quantization=None (full-precision path)."""
    worker_calls: list[dict] = []

    def fake_from_path(cls_or_path, max_seq_len=2048, quantization=None, **kwargs):
        worker_calls.append({"quantization": quantization})
        return InferenceWorker.from_engine(_FakeEngine())

    monkeypatch.setattr(InferenceWorker, "from_path", fake_from_path)

    create_app_from_path(Path("/fake/model"))

    assert worker_calls[0]["quantization"] is None


def test_worker_from_path_raises_on_engine_load_failure(monkeypatch):
    """from_path() must propagate engine load exceptions instead of hanging.

    When Engine.from_model_path raises (e.g., incompatible group_size), the
    worker thread sets the ready event so the caller is never blocked, and
    from_path() re-raises the original exception on the caller thread.
    """
    def fake_from_model_path(model_path, max_seq_len=2048, quantization=None):
        raise ValueError("in_features=48 not divisible by group_size=64")

    monkeypatch.setattr(Engine, "from_model_path", fake_from_model_path)

    with pytest.raises(ValueError, match="not divisible"):
        InferenceWorker.from_path("/fake/model", max_seq_len=64)
