"""
Tests for tiny_duo_infer.serving.api.

The server is tested via FastAPI's TestClient (httpx-backed sync client).
All unit tests use _FakeEngine so no model artifacts are required.

Streaming format: one NDJSON line per event.
  Fragment: {"done": false, "text": "<fragment>"}
  Final:    {"done": true, "text": "<full trimmed text>", "prompt_tokens": N,
             "generated_tokens": N, "stop_reason": "<reason>"}
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

import tiny_duo_infer.serving.api as api_module
from tiny_duo_infer.generation import GenerationRequest, GenerationResponse
from tiny_duo_infer.serving.api import create_app


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
    """Reset module-level engine and lock between tests."""
    api_module._engine = None
    if api_module._lock.locked():
        api_module._lock.release()
    yield
    if api_module._lock.locked():
        api_module._lock.release()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_engine_not_initialized_does_not_leak_lock():
    """Requests before create_app() fail cleanly without leaving the lock held."""
    from tiny_duo_infer.serving.api import app as _app

    api_module._engine = None
    client = TestClient(_app, raise_server_exceptions=False)

    # First request: engine not initialised → 500
    resp1 = client.post("/generate", json={"prompt": "hi"})
    assert resp1.status_code == 500
    assert not api_module._lock.locked(), "lock must be released after engine error"

    # Second request: also returns 500, not 503 (lock was not leaked)
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


def test_health_shows_active_when_locked():
    """GET /health reports active=True while the lock is held."""
    client = TestClient(create_app(_FakeEngine()))
    api_module._lock.acquire()
    try:
        resp = client.get("/health")
        assert resp.json()["active"] is True
    finally:
        api_module._lock.release()


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
    """POST /generate against real Qwen3 model completes and returns metadata."""
    import os
    from pathlib import Path
    from tiny_duo_infer.engine import Engine

    model_path = os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b")
    engine = Engine.from_model_path(Path(model_path), max_seq_len=512)
    client = TestClient(create_app(engine))

    resp = client.post(
        "/generate",
        json={"prompt": "The capital of France is", "max_new_tokens": 2, "temperature": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["text"], str)
    assert body["stop_reason"] in ("eos", "max_new_tokens", "context_length")


@pytest.mark.slow
def test_qwen3_generate_stream_smoke():
    """POST /generate/stream against real Qwen3 model returns valid NDJSON."""
    import os
    from pathlib import Path
    from tiny_duo_infer.engine import Engine

    model_path = os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b")
    engine = Engine.from_model_path(Path(model_path), max_seq_len=512)
    client = TestClient(create_app(engine))

    resp = client.post(
        "/generate/stream",
        json={"prompt": "Hello", "max_new_tokens": 2, "temperature": 0.0},
    )
    assert resp.status_code == 200
    chunks = _parse_ndjson(resp.text)
    assert chunks[-1]["done"] is True
    assert chunks[-1]["stop_reason"] in ("eos", "max_new_tokens", "context_length")
