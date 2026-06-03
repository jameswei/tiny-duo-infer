"""
Single-request HTTP inference server.

Exposes three endpoints over FastAPI:

  GET  /health           — server liveness and busy status.
  POST /generate         — full-response generation (JSON).
  POST /generate/stream  — streaming generation (NDJSON).

The server loads one model per process and serializes requests through a
single InferenceWorker thread.  Concurrent requests receive a 503 "server
busy" response rather than waiting in a queue.

Engine lifecycle separation:
  - CLI: engine is owned directly by the CLI process (no HTTP layer).
  - HTTP server: engine is owned by InferenceWorker, which initialises it
    inside its dedicated thread so that all MLX GPU operations stay on that
    thread (Apple Silicon GPU stream thread-affinity requirement).

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

Entrypoints:
    # Production (engine initialised inside worker thread):
    uv run python -m tiny_duo_infer.serving.api \\
      --model-path ./models/qwen3-0.6b \\
      --max-seq-len 2048

    # Testing (wrap a pre-built engine):
    create_app(fake_engine)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tiny_duo_infer.generation import ChatMessage, GenerationRequest, GenerationResponse
from tiny_duo_infer.serving.worker import InferenceWorker


_worker: InferenceWorker | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Worker is created before uvicorn.run() via create_app_from_path();
    # nothing to do on startup.
    yield
    # Shut down the worker on server stop so the inference thread exits cleanly.
    if _worker is not None:
        _worker.shutdown()


app = FastAPI(
    title="tiny-duo-infer",
    description="Single-request local LLM inference server.",
    lifespan=_lifespan,
)


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


def _require_worker() -> InferenceWorker:
    if _worker is None:
        raise RuntimeError(
            "inference worker not initialised; "
            "call create_app() or create_app_from_path() first"
        )
    return _worker


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server liveness and whether a generation request is active."""
    active = _worker.busy if _worker is not None else False
    return HealthResponse(status="ok", active=active)


@app.post("/generate", response_model=GenerateResponseBody)
async def generate(body: GenerateRequestBody) -> GenerateResponseBody:
    """
    Run a full generation request and return the complete response as JSON.

    Returns 503 if another request is currently being processed.
    Returns 422 if request fields fail GenerationRequest validation.
    Returns 500 if the server was not initialised.
    """
    worker = _require_worker()

    try:
        request = _to_generation_request(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    loop = asyncio.get_event_loop()
    future: asyncio.Future[GenerationResponse] = loop.create_future()

    if not worker.submit_generate(request, future, loop):
        raise HTTPException(status_code=503, detail="server busy")

    response = await future
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
    Returns 500 if the server was not initialised.
    """
    worker = _require_worker()

    try:
        request = _to_generation_request(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    loop = asyncio.get_event_loop()
    item_queue: asyncio.Queue[Any] = asyncio.Queue()

    if not worker.submit_stream(request, item_queue, loop):
        raise HTTPException(status_code=503, detail="server busy")

    async def _iter_ndjson():
        while True:
            item = await item_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
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

    return StreamingResponse(_iter_ndjson(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# App factories
# ---------------------------------------------------------------------------


def create_app(engine: Any) -> FastAPI:
    """Wrap a pre-built engine and return the configured app.

    Intended for unit tests that supply a fake engine.  The engine is wrapped
    in an InferenceWorker so routing is identical to the production path.
    """
    global _worker
    _worker = InferenceWorker.from_engine(engine)
    return app


def create_app_from_path(model_path: Path, max_seq_len: int = 2048) -> FastAPI:
    """Load the engine inside the inference worker thread and return the app.

    Use this for production and slow smoke tests.  The engine is initialised
    on the worker thread so all MLX GPU operations remain on that thread
    (Apple Silicon GPU stream thread-affinity requirement).
    """
    global _worker
    _worker = InferenceWorker.from_path(model_path, max_seq_len)
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

    create_app_from_path(Path(args.model_path), max_seq_len=args.max_seq_len)
    uvicorn.run(app, host=args.host, port=args.port)
