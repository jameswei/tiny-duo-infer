"""
Single-request HTTP inference server.

Exposes three endpoints over FastAPI:

  GET  /health           — server liveness and busy status.
  POST /generate         — full-response generation (JSON).
  POST /generate/stream  — streaming generation (NDJSON).

The server loads one model per process and serializes requests with a
threading.Lock. Concurrent requests receive a 503 "server busy" response
rather than waiting in a queue.

Streaming format (NDJSON, one JSON object per line):

  Fragment chunk:  {"done": false, "text": "<fragment>"}
  Final chunk:     {"done": true, "text": "<full trimmed text>",
                    "prompt_tokens": N, "generated_tokens": N,
                    "stop_reason": "<reason>"}

Stop-string edge case for streaming: when a stop string spans a token boundary
the fragment containing the marker boundary may already be sent before the
match is confirmed. The final chunk always carries the correctly trimmed text
(as a convenience copy); streaming callers that need strict trimming should use
POST /generate instead.

Entrypoint:
    uv run python -m tiny_duo_infer.serving.api \\
      --model-path ./models/qwen3-0.6b \\
      --max-seq-len 2048
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tiny_duo_infer.engine import Engine
from tiny_duo_infer.generation import ChatMessage, GenerationRequest, GenerationResponse


app = FastAPI(
    title="tiny-duo-infer",
    description="Single-request local LLM inference server.",
)

_engine: Engine | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class _ChatMessageBody(BaseModel):
    role: str
    content: str


class GenerateRequestBody(BaseModel):
    prompt: str | None = None
    messages: list[_ChatMessageBody] | None = None
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stop: list[str] = []
    seed: int | None = None
    chat: bool = False


class GenerateResponseBody(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    stop_reason: str


class HealthResponse(BaseModel):
    status: str
    active: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_generation_request(body: GenerateRequestBody) -> GenerationRequest:
    msgs = None
    if body.messages is not None:
        msgs = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    return GenerationRequest(
        prompt=body.prompt,
        messages=msgs,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        stop=list(body.stop),
        seed=body.seed,
        chat=body.chat,
    )


def _require_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("engine not initialized; call create_app(engine) first")
    return _engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server liveness and whether a generation request is active."""
    return HealthResponse(status="ok", active=_lock.locked())


@app.post("/generate", response_model=GenerateResponseBody)
async def generate(body: GenerateRequestBody) -> GenerateResponseBody:
    """
    Run a full generation request and return the complete response as JSON.

    Returns 503 if another request is currently being processed.
    Returns 422 if request fields fail GenerationRequest validation.
    Returns 500 if the server was not initialised via create_app().
    """
    # Resolve engine before acquiring the lock so a missing engine never
    # leaves the server in a permanently-busy state.
    engine = _require_engine()
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="server busy")
    try:
        request = _to_generation_request(body)
    except ValueError as exc:
        _lock.release()
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        loop = asyncio.get_running_loop()
        response: GenerationResponse = await loop.run_in_executor(
            None, engine.generate_request, request
        )
    finally:
        _lock.release()

    return GenerateResponseBody(
        text=response.text,
        prompt_tokens=response.prompt_tokens,
        generated_tokens=response.generated_tokens,
        stop_reason=response.stop_reason,
    )


@app.post("/generate/stream")
async def generate_stream(body: GenerateRequestBody) -> StreamingResponse:
    """
    Run generation and stream decoded fragments as NDJSON.

    Each line of the response body is a complete JSON object:
      Fragment: {"done": false, "text": "<fragment>"}
      Final:    {"done": true, "text": "<full trimmed text>",
                 "prompt_tokens": N, "generated_tokens": N,
                 "stop_reason": "<reason>"}

    The "text" field of the final chunk carries the full accumulated text
    (stop-string trimmed) matching what POST /generate would return.

    Returns 503 if another request is currently being processed.
    Returns 422 if request fields fail GenerationRequest validation.
    Returns 500 if the server was not initialised via create_app().
    """
    # Resolve engine before acquiring the lock so a missing engine never
    # leaves the server in a permanently-busy state.
    engine = _require_engine()
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="server busy")
    try:
        request = _to_generation_request(body)
    except ValueError as exc:
        _lock.release()
        raise HTTPException(status_code=422, detail=str(exc))

    def _iter_ndjson() -> Iterator[str]:
        try:
            for item in engine.generate_stream(request):
                if isinstance(item, GenerationResponse):
                    yield json.dumps({
                        "done": True,
                        "text": item.text,
                        "prompt_tokens": item.prompt_tokens,
                        "generated_tokens": item.generated_tokens,
                        "stop_reason": item.stop_reason,
                    }) + "\n"
                else:
                    yield json.dumps({"done": False, "text": item}) + "\n"
        finally:
            _lock.release()

    return StreamingResponse(_iter_ndjson(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# App factory (used by tests and the CLI entrypoint)
# ---------------------------------------------------------------------------


def create_app(engine: Engine) -> FastAPI:
    """Bind a loaded engine to the server and return the configured app."""
    global _engine
    _engine = engine
    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="tiny_duo_infer.serving.api",
        description="Run the tiny-duo-infer HTTP inference server.",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to a local HuggingFace-compatible model directory.",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Maximum prompt + generated sequence length.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    args = parser.parse_args()

    _loaded = Engine.from_model_path(Path(args.model_path), max_seq_len=args.max_seq_len)
    create_app(_loaded)
    uvicorn.run(app, host=args.host, port=args.port)
