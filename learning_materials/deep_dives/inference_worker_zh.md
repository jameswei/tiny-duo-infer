# 深入解析：推理 Worker —— Async HTTP 背后的单线程 MLX

本文聚焦 `tiny-duo-infer` 在 Phase 1.6 加入的**推理 worker**。配套阅读：源码 `tiny_duo_infer/serving/worker.py` 与 `tiny_duo_infer/serving/api.py`、单元测试 `tests/test_serving.py`，以及 `tiny_duo_infer/engine.py` 中的引擎入口。

Phase 1.6 的目标不是写一个生产级 HTTP 服务器，而是把两个不兼容的执行模型之间最小的桥接显式化：

- **引擎一侧**：单线程持有 MLX GPU stream，每次只跑一条生成（`Engine.generate_request`、`Engine.generate_stream`）。
- **HTTP 一侧**：由 FastAPI/uvicorn 驱动的 `asyncio` 事件循环，必须保持非阻塞、能接收多条并发连接。

它们之间的桥就是 `InferenceWorker`。一旦理解它，你也就理解了为什么所有生产推理引擎最终都会长出一个"worker 线程"或"engine 进程"抽象。

---

## 1. 为什么必须有 worker 线程

MLX 在 Apple Silicon 上有一条硬规则：

> 针对某个 GPU stream 的所有操作，必须在**初始化该 stream 的那个线程**上执行。

如果引擎在线程 A 上构造、请求却在线程 B 上跑，MLX 要么直接报错，要么悄悄给出错误结果。FastAPI/uvicorn 把每个请求交给 asyncio 事件循环抽出的 worker task 处理，事件循环本身又跑在框架选定的线程上——**两条请求落在同一线程上**这件事**没有任何保证**。

满足 MLX 这条规则的最简设计是：

1. 把引擎以及它所有的 MLX 工作钉到**一个**专属后台线程，由 `InferenceWorker` 拥有。
2. 让 HTTP 路由处理函数留在 asyncio 事件循环里。
3. 通过线程安全队列把工作从事件循环投到 worker 线程，再用 `loop.call_soon_threadsafe()` 把结果投回事件循环。

两条并发请求会同时到达 worker 队列，但 worker 在同一线程上一次处理一条。这样既保留了 MLX 的规则，又**完全避免**从事件循环里调用 MLX。

```text
          asyncio 事件循环                       InferenceWorker 线程
          ─────────────────                      ─────────────────────
HTTP req ──> route handler ──> submit_generate    ─┐
HTTP req ──> route handler ──> submit_generate    ─┤  queue.Queue ──> _run() ──> Engine.generate_request
                                                  ─┘                                  │
            ◄── future.set_result ◄── call_soon_threadsafe ◄────────────────────────────┘
```

心智模型是：**两个执行域，一条队列在中间，两个方向各有专属 handoff 原语。**

---

## 2. 两条构造路径

`InferenceWorker` 暴露两个工厂 classmethod。两者最终都进入同一条 `_run()` 循环，但**引擎在哪里被构造**不同。

### `from_path` —— 生产路径

```python
worker = InferenceWorker.from_path(
    Path("./models/qwen3-0.6b"),
    max_seq_len=2048,
    quantization=QuantizationConfig(bits=4, group_size=64),
)
```

发生的事情：

1. 调用方创建一个 `InferenceWorker` 实例。
2. `_start(load_in_thread=True)` 起一个后台线程。
3. 在新线程内，`_run()` 调用 `Engine.from_model_path(model_path, max_seq_len, quantization=...)`。
4. 引擎——以及它所有的 MLX 数组、KV cache、量化权重、`mx.eval()` 边界——都在这个线程上初始化。
5. 工厂用一个 `threading.Event` 阻塞调用方，直到加载成功或失败。失败会在调用方线程上重新抛出，让一台配置错的服务器**在 uvicorn 绑定端口之前**就失败。

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

### `from_engine` —— 测试路径

```python
worker = InferenceWorker.from_engine(_FakeEngine())
```

把一个预构造好的引擎（通常是 fake）包起来，全程不触发 MLX。worker 线程仍然起，但 `_run()` 跳过加载分支。这样**单元测试与生产**的"路由 + 队列"代码路径完全一致——测试失败真的跑通的是 worker 逻辑，而不是某个并行的 mock 实现。

这种切分有意义：`from_engine` 的存在**仅仅**是因为没有 MLX 或模型 artifact 的 CI 机器无法启动真引擎。`tests/test_serving.py` 几乎所有测试都用 `from_engine`，慢路径 `from_path` 留给真模型 smoke。

---

## 3. Worker 主循环

线程一旦启动，就坐在最朴素的循环里：

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

三个性质让这条简单循环就够用：

- **一条队列、无优先级。** FIFO 意味着请求到达 worker 的顺序就是执行顺序。没有 scheduler，因为 Phase 1.6 把在飞请求数限制为 1（见 §5）。
- **任务是零参数 callable。** 每个入队项都是一个 closure，已经捕获了它的 `request`、`future` 和 `loop`——worker 线程**永远不**检查请求形态，因此它不需要知道 `GenerationRequest` 或 HTTP 语义。
- **`None` 是关闭哨兵。** `worker.shutdown()` 投一个 `None` 然后 join 线程。没有 flag、没有信号、没有靠 daemon 线程"被遗弃"。**关闭是确定性的。**

两个工厂构造的 `_run` 入口条件不同，但循环本体完全一样。

---

## 4. 两套提交接口：`submit_generate` 与 `submit_stream`

桥接原语对方向敏感。非流式请求只产生一个结果；流式请求会产出多段 fragment 加一个最终结果。worker 用两个不同方法暴露它们，让协议**没有歧义**。

### `submit_generate` —— 通过 `asyncio.Future` 给单个结果

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

HTTP 路由 `await` 这个 future：

```python
# tiny_duo_infer/serving/api.py
loop = asyncio.get_event_loop()
future = loop.create_future()
if not worker.submit_generate(request, future, loop):
    raise HTTPException(status_code=503, detail="server busy")
response = await future
```

把结果送回事件循环用的是 `loop.call_soon_threadsafe`——这是从**非事件循环线程**与 asyncio loop 交互**唯一**有文档保证的方式。直接在 worker 线程上调用 `future.set_result(...)` 会与事件循环自身的簿记产生竞态，是 undefined behaviour。

### `submit_stream` —— 通过 `asyncio.Queue` 加哨兵给多段 fragment

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

队列里运送三类消息：

| 项类型 | 含义 |
|---|---|
| `str` | 来自 `Engine.generate_stream` 的解码 fragment |
| `GenerationResponse` | 包含完整文本与 stats 的最终响应 |
| `Exception` | 引擎抛出；消费方必须在事件循环侧重新抛出 |
| `None` | 哨兵：流已干净结束 |

HTTP 路由读队列，按行输出 NDJSON：

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

`None` 哨兵是流式响应能干净关闭的关键——没有它，消费方会在下一次 `await item_queue.get()` 上**永远阻塞**。

> [!IMPORTANT]
> `_task` 中的 `finally` 块在成功路径上会**先**清除 `_busy_event`，**再**入队 `None` 哨兵。这是有意为之：引擎一结束就能接受下一条请求，**即使消费方还没把 asyncio 队列里最后几段 fragment 取走**。

---

## 5. 单请求在飞语义

Phase 1.6 强制每台服务一次只处理一条请求。机制是两个原语协作：

```python
# tiny_duo_infer/serving/worker.py
self._busy_event = threading.Event()   # 任意线程都能观察
self._busy_mutex = threading.Lock()    # 保护 test-and-set
```

两个提交接口里的 acquire 模式都是：

```python
with self._busy_mutex:
    if self._busy_event.is_set():
        return False        # 调用方把它转成 HTTP 503
    self._busy_event.set()
```

为什么两者都要？**`Event` 单独不能做安全的 test-and-set**：两条并发调用都可能看到 `is_set() == False`，然后两者都执行 `set()`。`Lock` 把 test-and-set 串行化；`Event` 是可被 `worker.busy` 与 `/health` 端点查询的**可观察状态**。

`_task` 的释放路径：

```python
finally:
    self._busy_event.clear()
```

`finally` 保证：即使引擎抛错、即使消费方半途取消流式 HTTP 连接，槽位也会被释放。**没有 `finally`，一次失败请求就会让服务器永久"忙"，拒绝后续每一条请求。**

HTTP 层把 `False` 返回值翻译成 HTTP 503：

```python
# tiny_duo_infer/serving/api.py
if not worker.submit_generate(request, future, loop):
    raise HTTPException(status_code=503, detail="server busy")
```

503（"Service Unavailable"）在这里是正确状态码：请求未入队，客户端应当重试。系统**不**内置客户端重试——这是 Phase 1.6 有意做的简化，留给真正引入"多请求调度"的 phase 来解决。

---

## 6. 错误传播

错误跨越线程边界有两个方向，每个方向都有专属信道：

| 来源 | 信道 | 表面 |
|---|---|---|
| 引擎加载期失败（模型缺失、config 非法等） | `_load_error` 字段，然后在调用方线程上重新抛出 | `from_path()` 在 `uvicorn.run()` 之前抛出 |
| 请求中引擎运行时失败 | `loop.call_soon_threadsafe(future.set_exception, exc)` 或 `item_queue.put_nowait(exc)` | HTTP 路由的 `await future` 重新抛出；流式由 NDJSON 消费方重新抛出 |
| 请求体非法 | `_to_generation_request` 抛 `ValueError`；路由返回 HTTP 422 | 校验**永远到不了** worker |

三条路径汇聚到同一条干净的规则：**worker 线程绝不静默吞掉异常。** 要么调用方在它自己的线程上看到，要么服务器启动失败。

`from_path` 的错误路径值得单拿出来强调：

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

加载失败路径**永远**在返回前 `ready.set()`，因此 `_start()` 中的调用方永远不会在 `ready.wait()` 上死等。`ready.wait()` 返回后，工厂检查 `_load_error` 并在主线程重新抛出。这条模式——"在汇报失败之前先 set rendezvous event"——正是阻止"半初始化的服务器把测试挂死"的关键。

---

## 7. 一次流式请求的完整生命周期

把所有部件拼起来，一条 `POST /generate/stream` 请求的完整生命：

```text
Client            Event Loop                Worker Thread             Engine
──────            ──────────                ─────────────             ──────

POST /generate/stream
  ──────────────►
                  校验 body (422?)
                  构造 GenerationRequest
                  busy? → yes ⇒ HTTP 503; no ⇒ enqueue _task
                  ──────────────────────►
                                            出队 _task
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
                  ◄── response ── 然后 ── None
                  yield NDJSON {"done":true,"text":..,"stats":..}
                  StreamingResponse 关闭 body
  ◄──────────────
```

从这张图必须内化两个细节：

- `busy_event.clear()` 发生在最终响应与哨兵入队**之前**。这就是"引擎一返回，下一条请求就能被接受、而不必等客户端读完"的**唯一原因**。
- asyncio 队列充当跨线程流式原语。**worker 线程不知道连接是 HTTP**；**路由处理函数不知道生产者是 MLX**。任意一边都可被替换——例如把 HTTP 换成 CLI 交互式 REPL——而 worker 完全不变。

---

## 8. 生命周期与测试钩子

worker 之上有两个 FastAPI 工厂：

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

二者都走 `_lifespan` context manager，让 uvicorn 退出时 `worker.shutdown()` 干净执行：

```python
@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    if _worker is not None:
        _worker.shutdown()
```

测试惯用模式是 `create_app(_FakeEngine())`，再用 `fastapi.testclient.TestClient` 驱动这个 FastAPI app。因为 worker 是真的、引擎是 fake，测试覆盖了：入队、busy 串行化、asyncio handoff、NDJSON 格式化、shutdown。它们**不**覆盖 MLX、模型 artifact、生成正确性——那些归 Phase 1 的引擎测试。

---

## 9. 阅读代码时要写下的清单

针对 worker 子系统，写下：

- **输入：** 通过 `submit_generate` / `submit_stream` 传入的 `GenerationRequest`，以及 asyncio `loop` 与一个 `asyncio.Future` 或 `asyncio.Queue`。
- **输出：** future 上的一个结果，或者推进队列的若干项；两种情况下提交方法都返回 `bool` 表示是否被入队。
- **张量形状：** 无——worker 与引擎无关；张量只存在于引擎调用内部。
- **状态：**
  - `_engine`（仅 `ready.set()` 返回后可在 worker 线程上访问）
  - `_task_queue`（跨线程）
  - `_busy_event` 与 `_busy_mutex`（跨线程）
  - `_load_error`（worker 写、调用方读）
- **不变量：**
  - 引擎只在 worker 线程上被触碰。
  - `_busy_event` 在 mutex 内被 set，在任务的 `finally` 里被 clear。
  - `loop.call_soon_threadsafe` 是 worker 到事件循环的**唯一**调用。
  - 流式成功以一个 `None` 哨兵结束；流式失败以一个 `Exception` 结束（**不**再追加哨兵）。
- **失败情况：** 模型加载失败（由 `from_path` 抛出）、请求中引擎抛错（通过 future 或队列投递）、busy 拒绝（调用方转成 HTTP 503）、请求体非法（HTTP 422，根本不会进 worker）。
- **静默生成 corruption 的危险点：** 把任何 MLX 调用从 worker 线程**搬出去**——例如在 asyncio loop 上直接调 `mx.eval(...)`、或在主线程上构造引擎。生成仍会输出数字，但数字来自一个**未被调用线程初始化的 stream**——MLX 要么报错，要么更糟，悄悄成功但 KV-cache 状态错误。

---

## 10. 与生产引擎的距离：本设计止步于何处

Phase 1.6 设计是有意做边界的：

- **一条在飞请求。** 生产引擎接受多条并发请求并用 continuous batching 调度。那是 Phase 3 的工作（`docs/phases/README.md` —— Phase 1.10 / 3 行）。
- **单进程。** 真实部署在多 worker 进程之上挂负载均衡器；本引擎没有这个。
- **没有抢占 / 取消。** 一次长生成无法中途取消；asyncio 那一侧只能等。
- **除了 503 之外没有反压。** `submit_stream` 内的 asyncio 队列是无界的——理论上慢客户端可让 fragment 累积速度超过排空速度。Phase 1.6 不限其上界，因为 worker 是单请求在飞，队列最多深度等于一次生成。

每条限制都映射到一个目前是"directional"而非 active 的 phase。这台最小 worker 的意义是：**在尝试解决这些 Phase 3 trade-off 之前，先让它们可见**。continuous batching 必须把任务调度到**恰好一个**MLX-持有线程上——这里的抽象就是那个 scheduler 的最小版本。

---

## 延伸阅读

- `docs/phases/phase-1.6-generation-serving.md`：phase 权威 spec。
- `tiny_duo_infer/serving/worker.py`：`InferenceWorker`、`from_path`、`from_engine`、`submit_generate`、`submit_stream`、`shutdown`。
- `tiny_duo_infer/serving/api.py`：FastAPI 路由、`_to_generation_request`、NDJSON 流、`_lifespan`。
- `tiny_duo_infer/engine.py`：`Engine.generate_request`、`Engine.generate_stream`、`Engine.from_model_path`（含 Phase 1.8 的 `quantization=` 转发）。
- `tests/test_serving.py`：fake-engine 接线、busy/503、NDJSON 解析、错误传播、shutdown 行为。
- `learning_materials/roadmap.md`：引导阅读顺序。
- `learning_materials/deep_dives/chat_templating.md`：生成本 worker 实际送进引擎的那串 prompt 字符串的格式化器。
