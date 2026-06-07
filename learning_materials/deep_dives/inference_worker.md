# Deep Dive: The Inference Worker — Single-Threaded MLX Behind Async HTTP

This document is a focused walkthrough of the **inference worker** added in
Phase 1.6 of `tiny-duo-infer`. It pairs with the source files
`tiny_duo_infer/serving/worker.py` and `tiny_duo_infer/serving/api.py`, the
unit tests in `tests/test_serving.py`, and the engine entry points in
`tiny_duo_infer/engine.py`.

The Phase 1.6 goal is not a production HTTP server. The goal is to make
visible the smallest possible bridge between two incompatible execution
models:

- **The engine side**: a single thread that owns the MLX GPU stream and runs
  one generation at a time (`Engine.generate_request`, `Engine.generate_stream`).
- **The HTTP side**: an `asyncio` event loop driven by FastAPI/uvicorn that
  must remain non-blocking and accept many concurrent connections.

The bridge between them is `InferenceWorker`. Once you understand it, you
also understand why every production inference engine eventually grows a
"worker thread" or "engine process" abstraction.

---

## 1. Why A Worker Thread At All

MLX has one hard rule on Apple Silicon:

> All operations against a given GPU stream must run on the thread that
> initialised that stream.

If the engine is constructed on thread A and a request runs on thread B, MLX
will either error out or silently produce wrong results. FastAPI/uvicorn
handle each request on a worker task drawn from an asyncio event loop, which
in turn runs on a thread chosen by the framework — there is no guarantee
that two requests share the same thread.

The simplest design that satisfies the MLX rule is:

1. Pin the engine and all of its MLX work to **one** dedicated background
   thread, owned by `InferenceWorker`.
2. Let HTTP request handlers stay on the asyncio event loop.
3. Submit work from the event loop to the worker thread via a thread-safe
   queue, and resolve the result back via `loop.call_soon_threadsafe()`.

Two concurrent requests will arrive at the worker queue, but the worker
processes them one at a time on the same thread. The MLX rule is preserved
without ever calling MLX from the event loop.

```text
          asyncio event loop                      InferenceWorker thread
          ─────────────────                       ─────────────────────
HTTP req ──> route handler ──> submit_generate    ─┐
HTTP req ──> route handler ──> submit_generate    ─┤  queue.Queue ──> _run() ──> Engine.generate_request
                                                  ─┘                                  │
            ◄── future.set_result ◄── call_soon_threadsafe ◄────────────────────────────┘
```

The mental model is: **two execution domains, one queue between them, two
direction-specific handoff primitives.**

---

## 2. The Two Construction Paths

`InferenceWorker` exposes two factory classmethods. Both end up at the same
`_run()` loop, but they differ on *where* the engine is built.

### `from_path` — production

```python
worker = InferenceWorker.from_path(
    Path("./models/qwen3-0.6b"),
    max_seq_len=2048,
    quantization=QuantizationConfig(bits=4, group_size=64),
)
```

What happens:

1. The caller creates an `InferenceWorker` instance.
2. `_start(load_in_thread=True)` spawns the background thread.
3. Inside the new thread, `_run()` calls
   `Engine.from_model_path(model_path, max_seq_len, quantization=...)`.
4. The engine — and therefore every MLX array, KV cache, quantized weight,
   and `mx.eval()` boundary — is initialised on that thread.
5. The factory blocks the caller on a `threading.Event` until either
   loading succeeds or fails; failures are re-raised on the caller's thread
   so a misconfigured server fails before uvicorn ever binds the port.

```python
# tiny_duo_infer/serving/worker.py
@classmethod
def from_path(cls, model_path, max_seq_len, quantization=None):
    worker = cls()
    worker._model_path = model_path
    worker._max_seq_len = max_seq_len
    worker._quantization = quantization
    worker._start(load_in_thread=True)
    if worker._load_error is not None:
        raise worker._load_error
    return worker
```

### `from_engine` — testing

```python
worker = InferenceWorker.from_engine(_FakeEngine())
```

A pre-built engine (typically a fake) is wrapped without touching MLX. The
worker thread still spins up, but `_run()` skips the loading branch. This
keeps the routing and queueing code path identical between unit tests and
production, so test failures actually exercise the worker logic and not a
parallel mock implementation.

The split matters: `from_engine` exists *only* because real-engine startup
on a CI machine without MLX or model artifacts would be impossible.
`tests/test_serving.py` uses `from_engine` for nearly every test, and the
slow path through `from_path` is reserved for real-model smokes.

---

## 3. The Worker Loop

Once running, the thread sits in the simplest possible loop:

```python
# tiny_duo_infer/serving/worker.py
def _run(self, load_in_thread, ready):
    if load_in_thread:
        try:
            self._engine = Engine.from_model_path(...)
        except Exception as exc:
            self._load_error = exc
            ready.set()
            return
    ready.set()

    while True:
        task = self._task_queue.get()
        if task is None:
            break
        task()
```

Three properties make this simple loop sufficient:

- **One queue, no priority.** FIFO means the order requests reach the worker
  is the order they execute. There is no scheduler, because Phase 1.6 caps
  in-flight requests at one (see §5).
- **Tasks are zero-argument callables.** Every queued item is a closure that
  already captured its `request`, `future`, and `loop` — the worker thread
  never inspects the request shape, so it does not need to know about
  `GenerationRequest` or HTTP semantics.
- **`None` is the shutdown sentinel.** `worker.shutdown()` puts a single
  `None` and joins the thread. There is no flag, no signal, no daemon-thread
  abandonment. Shutdown is deterministic.

The two factories build *different* `_run` entry conditions, but the loop
itself is the same.

---

## 4. Two Submission APIs: `submit_generate` And `submit_stream`

The bridge primitive is direction-specific. A non-streaming request returns
exactly one result; a streaming request emits many fragments plus a final
result. The worker exposes both as separate methods to keep the protocol
unambiguous.

### `submit_generate` — single result via `asyncio.Future`

```python
# tiny_duo_infer/serving/worker.py
def submit_generate(self, request, future, loop) -> bool:
    with self._busy_mutex:
        if self._busy_event.is_set():
            return False
        self._busy_event.set()

    def _task():
        try:
            response = self._engine.generate_request(request)
            loop.call_soon_threadsafe(future.set_result, response)
        except Exception as exc:
            loop.call_soon_threadsafe(future.set_exception, exc)
        finally:
            self._busy_event.clear()

    self._task_queue.put(_task)
    return True
```

The HTTP route awaits the future:

```python
# tiny_duo_infer/serving/api.py
loop = asyncio.get_event_loop()
future = loop.create_future()
if not worker.submit_generate(request, future, loop):
    raise HTTPException(status_code=503, detail="server busy")
response = await future
```

The handoff back to the event loop is `loop.call_soon_threadsafe`, which is
the *only* documented way to interact with an asyncio loop from a non-loop
thread. Calling `future.set_result(...)` directly from the worker thread
would race with the loop's own bookkeeping and is undefined behaviour.

### `submit_stream` — many fragments via `asyncio.Queue` plus a sentinel

```python
def submit_stream(self, request, item_queue, loop) -> bool:
    with self._busy_mutex:
        if self._busy_event.is_set():
            return False
        self._busy_event.set()

    def _task():
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
```

The queue carries three message classes:

| Item type | Meaning |
|---|---|
| `str` | a decoded fragment from `Engine.generate_stream` |
| `GenerationResponse` | the final response with full text and stats |
| `Exception` | the engine raised; the consumer must re-raise on the loop side |
| `None` | sentinel: the stream has ended cleanly |

The HTTP route reads the queue and emits NDJSON lines:

```python
# tiny_duo_infer/serving/api.py
async def _iter_ndjson():
    while True:
        item = await item_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        if isinstance(item, GenerationResponse):
            yield json.dumps({"done": True, "text": item.text, ...}) + "\n"
        else:
            yield json.dumps({"done": False, "text": item}) + "\n"
```

The `None` sentinel is what closes the streaming response cleanly. Without
it, the consumer would block forever on the next `await item_queue.get()`.

> [!IMPORTANT]
> The `finally` block in `_task` clears `_busy_event` *before* the `None`
> sentinel is enqueued in the success path. That is intentional: the next
> request can be accepted as soon as the engine finishes, even if the
> consumer hasn't yet drained the last few fragments off the asyncio queue.

---

## 5. Single-In-Flight Semantics

Phase 1.6 enforces one concurrent request per server. The mechanism is two
primitives working together:

```python
# tiny_duo_infer/serving/worker.py
self._busy_event = threading.Event()   # observable from any thread
self._busy_mutex = threading.Lock()    # protects the test-and-set
```

The acquire pattern in both submission APIs is:

```python
with self._busy_mutex:
    if self._busy_event.is_set():
        return False        # caller turns this into HTTP 503
    self._busy_event.set()
```

Why both? `Event` alone is not safe for test-and-set: two concurrent calls
could each see `is_set() == False` and both proceed to `set()`. The `Lock`
serialises that test-and-set; the `Event` is the *observable* state queried
by `worker.busy` and the `/health` endpoint.

The release path in `_task`:

```python
finally:
    self._busy_event.clear()
```

`finally` ensures the slot is released even if the engine raises, even if
the consumer cancels the streaming HTTP connection mid-flight. Without
`finally`, a single failed request would leave the server permanently
"busy" and refusing every subsequent request.

The HTTP layer translates a `False` return into HTTP 503:

```python
# tiny_duo_infer/serving/api.py
if not worker.submit_generate(request, future, loop):
    raise HTTPException(status_code=503, detail="server busy")
```

503 ("Service Unavailable") is the correct status here: the request was not
queued, the client should retry. There is no built-in client-side retry —
that is a deliberate Phase-1.6 simplification, deferred to whatever phase
introduces real multi-request scheduling.

---

## 6. Error Propagation

Errors cross the thread boundary in two directions, and each direction has a
dedicated channel:

| Source | Channel | Surface |
|---|---|---|
| Engine load-time failure (model missing, config invalid, etc.) | `_load_error` field, then re-raised on the caller thread | `from_path()` raises before `uvicorn.run()` is called |
| Engine runtime failure during a request | `loop.call_soon_threadsafe(future.set_exception, exc)` or `item_queue.put_nowait(exc)` | The HTTP route's `await future` re-raises; for streams the NDJSON consumer re-raises |
| Bad request body | `_to_generation_request` raises `ValueError`; the route returns HTTP 422 | Validation never reaches the worker |

All three paths converge on a clean rule: **the worker thread never silently
swallows an exception.** Either the caller sees it on their thread, or the
server fails to start.

The `from_path` error path is worth highlighting:

```python
# tiny_duo_infer/serving/worker.py
def _run(self, load_in_thread, ready):
    if load_in_thread:
        try:
            self._engine = Engine.from_model_path(...)
        except Exception as exc:
            self._load_error = exc
            ready.set()
            return
    ready.set()
    ...
```

The load failure path **always sets `ready`** before returning, so the caller
in `_start()` never blocks forever on `ready.wait()`. After `ready.wait()`
returns, the factory inspects `_load_error` and re-raises on the main thread.
This pattern — "set the rendezvous event before reporting failure" — is
specifically what prevents a half-initialised server from hanging tests.

---

## 7. The Lifespan Of A Streaming Request

Putting all the pieces together, here is the full life of one
`POST /generate/stream` call:

```text
Client            Event Loop                Worker Thread             Engine
──────            ──────────                ─────────────             ──────

POST /generate/stream
  ──────────────►
                  validate body (422?)
                  build GenerationRequest
                  busy? → yes ⇒ HTTP 503; no ⇒ enqueue _task
                  ──────────────────────►
                                            dequeue _task
                                            ───────────────────────►
                                                                    generate_stream
                                                                    yields "Hel"
                                            ◄───────────────────────
                  ◄── put_nowait("Hel") ── (call_soon_threadsafe)
                  yield NDJSON {"done":false,"text":"Hel"}
  ◄──────────────
                                                                    yields "lo"
                                            ◄───────────────────────
                  ◄── put_nowait("lo")
                  yield NDJSON {"done":false,"text":"lo"}
  ◄──────────────
                                                                    yields GenerationResponse
                                            ◄───────────────────────
                                            finally: busy_event.clear()
                                            put_nowait(response)
                                            put_nowait(None)
                  ◄── response ── then ── None
                  yield NDJSON {"done":true,"text":..,"stats":..}
                  StreamingResponse closes body
  ◄──────────────
```

Two details to internalise from this diagram:

- `busy_event.clear()` happens **before** the final response and sentinel are
  enqueued. That is the single-place reason why the next request can be
  accepted as soon as the engine returns, not when the client finishes
  reading.
- The asyncio queue is serving as the cross-thread streaming primitive. The
  worker thread does not know the connection is HTTP, and the route handler
  does not know the producer is MLX. Either side could be replaced — for
  example, swapping HTTP for a CLI interactive REPL — and the worker would
  not change.

---

## 8. Lifespan And Test Hooks

Two FastAPI factories sit on top of the worker:

```python
# tiny_duo_infer/serving/api.py
def create_app(engine):
    global _worker
    _worker = InferenceWorker.from_engine(engine)
    return app

def create_app_from_path(model_path, max_seq_len=2048, quantization=None):
    global _worker
    _worker = InferenceWorker.from_path(model_path, max_seq_len, quantization=quantization)
    return app
```

Both flow through the `_lifespan` context manager so `worker.shutdown()`
runs cleanly when uvicorn exits:

```python
@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    if _worker is not None:
        _worker.shutdown()
```

The test pattern is `create_app(_FakeEngine())`, then drive the resulting
FastAPI app with `fastapi.testclient.TestClient`. Because the worker is
real and the engine is fake, the tests exercise: queueing, busy serialisation,
asyncio handoff, NDJSON formatting, and shutdown. They do not exercise MLX,
model artifacts, or generation correctness — those live in Phase 1's engine
tests.

---

## 9. What To Verify When You Read The Code

For the worker subsystem, write down:

- **Inputs:** `GenerationRequest` over `submit_generate` / `submit_stream`,
  plus the asyncio `loop` and either an `asyncio.Future` or `asyncio.Queue`.
- **Outputs:** a result on the future, or items pushed onto the queue; in
  both cases `bool` from the submission method indicates whether the work
  was queued.
- **Tensor shapes:** none — the worker is engine-agnostic; tensors only
  exist inside the engine call.
- **State:**
  - `_engine` (worker thread only after `ready.set()` returns)
  - `_task_queue` (cross-thread)
  - `_busy_event` and `_busy_mutex` (cross-thread)
  - `_load_error` (set in worker, read in caller)
- **Invariants:**
  - The engine is touched only on the worker thread.
  - `_busy_event` is set inside the mutex and cleared in the task's `finally`.
  - `loop.call_soon_threadsafe` is the only worker-to-loop call.
  - Streaming success ends with a `None` sentinel; streaming failure ends
    with an `Exception` (and no sentinel).
- **Failure cases:** model load failure (raised by `from_path`), engine
  exception during a request (delivered via the future or queue), busy
  rejection (caller turns into HTTP 503), invalid request body (HTTP 422
  before the worker is involved).
- **The one thing that would silently corrupt things:** moving any MLX
  call out of the worker thread — for example, calling
  `mx.eval(...)` directly from the asyncio loop or building the engine
  on the main thread. Generation would still produce numbers, but the
  numbers would come from a stream that was not initialised by the
  calling thread, and either MLX raises, or worse, succeeds with
  silently incorrect KV-cache state.

---

## 10. Bridging To Production: Where This Design Stops

The Phase 1.6 design is intentionally bounded:

- **One in-flight request.** Production engines accept many concurrent
  requests and schedule them with continuous batching. That is Phase 3 work
  (`docs/phases/README.md` — Phase 1.10 / 3 lines).
- **One process.** Real deployments run multiple worker processes behind a
  load balancer; this engine has none.
- **No pre-empting / cancellation.** A long generation cannot be cancelled
  mid-flight; the asyncio side simply waits.
- **No backpressure beyond 503.** The asyncio queue inside `submit_stream`
  is unbounded; a slow client could in principle accumulate fragments faster
  than they drain. Phase 1.6 does not bound it because the worker is
  single-in-flight, so the queue is at most one generation deep.

Each of these limits maps to a Phase that is currently directional rather
than active. The point of this minimal worker is to make those Phase 3
trade-offs visible before trying to solve them: continuous batching needs to
schedule onto exactly one MLX-owning thread, and the abstraction here is the
smallest version of that scheduler.

---

## Further Reading

- `docs/phases/phase-1.6-generation-serving.md` — authoritative phase spec.
- `tiny_duo_infer/serving/worker.py` — `InferenceWorker`, `from_path`,
  `from_engine`, `submit_generate`, `submit_stream`, `shutdown`.
- `tiny_duo_infer/serving/api.py` — FastAPI routes, `_to_generation_request`,
  NDJSON streaming, `_lifespan`.
- `tiny_duo_infer/engine.py` — `Engine.generate_request`,
  `Engine.generate_stream`, `Engine.from_model_path` (with
  `quantization=` Phase 1.8 forwarding).
- `tests/test_serving.py` — fake-engine wiring, busy/503, NDJSON
  parsing, error propagation, shutdown behaviour.
- `learning_materials/roadmap.md` — guided reading order.
- `learning_materials/deep_dives/chat_templating.md` — the prompt formatter
  that produces the strings this worker sends to the engine.
