"""
Inference worker: a dedicated thread that owns the Engine and all MLX ops.

All MLX GPU operations must run on the thread that initialized the GPU stream.
InferenceWorker enforces this by owning a single background thread and routing
every engine call through it.

Two construction paths:
  from_path(model_path, max_seq_len)  — production; engine is created inside
      the worker thread, binding the MLX GPU stream to that thread.
  from_engine(engine)  — testing; wraps a pre-built (fake) engine that has no
      thread-affinity constraints.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path
from typing import Any

from tiny_duo_infer.generation import GenerationRequest, GenerationResponse
from tiny_duo_infer.quantization import QuantizationConfig


class InferenceWorker:
    """Single-threaded inference worker with async-friendly submit interface."""

    def __init__(self) -> None:
        self._task_queue: queue.Queue[Any] = queue.Queue()
        self._busy_event = threading.Event()
        self._busy_mutex = threading.Lock()
        self._thread: threading.Thread | None = None
        self._load_error: BaseException | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_engine(cls, engine: Any) -> InferenceWorker:
        """Wrap a pre-built engine (use for testing with fake engines)."""
        worker = cls()
        worker._engine = engine
        worker._start(load_in_thread=False)
        return worker

    @classmethod
    def from_path(
        cls,
        model_path: Path,
        max_seq_len: int,
        quantization: QuantizationConfig | None = None,
    ) -> InferenceWorker:
        """Create engine inside the worker thread (use for production).

        Blocks until the engine is fully loaded so the server is ready to
        accept requests as soon as this returns.

        Args:
            model_path:   path to a local HuggingFace-compatible model directory.
            max_seq_len:  maximum total sequence length (prompt + generated).
            quantization: optional weight-only quantization config (Phase 1.8).
                          Passed through to Engine.from_model_path() inside the
                          worker thread so MLX GPU stream affinity is preserved.
        """
        worker = cls()
        worker._model_path = model_path
        worker._max_seq_len = max_seq_len
        worker._quantization = quantization
        worker._start(load_in_thread=True)
        if worker._load_error is not None:
            raise worker._load_error
        return worker

    def _start(self, load_in_thread: bool) -> None:
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(load_in_thread, ready), daemon=True
        )
        self._thread.start()
        ready.wait()

    def _run(self, load_in_thread: bool, ready: threading.Event) -> None:
        if load_in_thread:
            from tiny_duo_infer.engine import Engine  # noqa: PLC0415
            try:
                self._engine = Engine.from_model_path(
                    self._model_path,
                    max_seq_len=self._max_seq_len,
                    quantization=self._quantization,
                )
            except Exception as exc:
                # Capture the error so from_path() can re-raise it on the
                # caller thread. Always set ready so the caller is never
                # left blocked in ready.wait().
                self._load_error = exc
                ready.set()
                return
        ready.set()

        while True:
            task = self._task_queue.get()
            if task is None:
                break
            task()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while a generation task is running."""
        return self._busy_event.is_set()

    def submit_generate(
        self,
        request: GenerationRequest,
        future: asyncio.Future,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Queue a generate_request task and resolve *future* with the result.

        Returns False (and does not queue) if a task is already running.
        The caller should return HTTP 503 in that case.
        """
        with self._busy_mutex:
            if self._busy_event.is_set():
                return False
            self._busy_event.set()

        def _task() -> None:
            try:
                response: GenerationResponse = self._engine.generate_request(request)
                loop.call_soon_threadsafe(future.set_result, response)
            except Exception as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            finally:
                self._busy_event.clear()

        self._task_queue.put(_task)
        return True

    def submit_stream(
        self,
        request: GenerationRequest,
        item_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Queue a generate_stream task and push items onto *item_queue*.

        Items are str fragments or a GenerationResponse final item, pushed via
        call_soon_threadsafe.  A None sentinel is pushed when the stream ends.
        If an exception occurs it is pushed in place of the sentinel.

        Returns False (and does not queue) if a task is already running.
        """
        with self._busy_mutex:
            if self._busy_event.is_set():
                return False
            self._busy_event.set()

        def _task() -> None:
            try:
                for item in self._engine.generate_stream(request):
                    loop.call_soon_threadsafe(item_queue.put_nowait, item)
            except Exception as exc:
                loop.call_soon_threadsafe(item_queue.put_nowait, exc)
                return
            finally:
                self._busy_event.clear()
            loop.call_soon_threadsafe(item_queue.put_nowait, None)

        self._task_queue.put(_task)
        return True

    def shutdown(self) -> None:
        """Stop the worker thread.  Call once at server shutdown."""
        self._task_queue.put(None)
        if self._thread is not None:
            self._thread.join()
