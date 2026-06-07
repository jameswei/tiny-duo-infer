# 深入解析：聊天模板与 ChatML 协议

本文聚焦 `tiny-duo-infer` 在 Phase 1.6 中加入的**聊天 prompt 协议**。配套阅读：源码 `tiny_duo_infer/prompt.py` 与 `tiny_duo_infer/generation.py`、单元测试 `tests/test_prompt.py`，以及 CLI / HTTP 入口 `tiny_duo_infer/cli.py` 与 `tiny_duo_infer/serving/api.py`。

Phase 1.6 的目标不是写一个通用模板引擎，而是把推理引擎与"聊天式调用方"之间的**契约**显式化：哪些特殊 token 必须出现、角色边界如何编码、为什么最后一个 assistant 回合要保持开放、哪些模型可以使用 chat 模式。

---

## 1. 为什么模板由引擎拥有

一个被微调过的 chat 模型期望 prompt 严格遵循它在微调阶段使用的字节级格式。两个边界必须被准确还原：

- **角色标记（Role markers）**：被 tokenizer 视为单一已注册单元的特殊 token。如果调用方发送的是普通文本（如 `"system: ..."`），tokenizer 会按裸字符编码——而模型从未把这视作角色边界。
- **回合分隔符（Turn separators）**：换行加上"回合结束"标记，用来告诉模型一个角色的内容到此为止、下一个角色开始了。

如果调用方自己组装这些边界，每个客户端都得学会模型训练时的具体格式。更糟糕的是，错配是**静默**的——模型仍然会输出"看起来像样"的东西，只是其实运行在错误的 prompt 上。

Phase 1.6 把这份契约固定在引擎一侧：

1. HTTP / CLI 层接收一份带结构的 `messages: list[ChatMessage]`，每条带 `role` 与 `content` 字段。
2. 引擎按 `config.model_type` 分发到 `tiny_duo_infer/prompt.py:format_chat_prompt` 中模型特定的格式化函数。
3. 输出是一个可以直接交给 `Tokenizer.encode(prompt_str, add_special_tokens=True)` 的字符串。

调用方永远看不到特殊 token；引擎也永远看不到错乱的 prompt。

---

## 2. ChatML 格式（Qwen3）

Qwen3 使用其 `tokenizer_config.json` 中的 **ChatML** 格式：

```text
<|im_start|>system
<system_content><|im_end|>
<|im_start|>user
<user_content><|im_end|>
<|im_start|>assistant
<assistant_content><|im_end|>
<|im_start|>user
<another_user_content><|im_end|>
<|im_start|>assistant
```

两条不可破坏的规则：

| 规则 | 原因 |
|---|---|
| `<|im_start|>` 与 `<|im_end|>` 是单一 token，且在 `tokenizer.json` 中以 special 注册。 | 如果没有 `AddedToken(special=True)` 注册，tokenizer 会把它们切成字节碎片，而 Qwen3 从未见过这种碎片。Roadmap §3 的 tokenizer 包装章节覆盖了这一查找路径。 |
| 最后那行 `<|im_start|>assistant\n` **不**用 `<|im_end|>` 关闭。 | 这是**生成提示**——保持 assistant 回合开放，等于告诉模型"你的任务是从这里继续生成"。如果用 `<|im_end|>` 关闭，等于说本回合已经结束、下一步该生成新角色标记，而不是 assistant 内容。 |

### tiny-duo-infer 的实现

```python
# tiny_duo_infer/prompt.py
def _format_chatml(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{_IM_START}{msg.role}\n{msg.content}{_IM_END}\n")
    parts.append(f"{_IM_START}assistant\n")
    return "".join(parts)
```

代码里没有"对历史 assistant 内容做特判"、没有"修剪行尾空白"、也没有"对 system 消息特殊处理"。整个格式就是：循环写出每条消息 + 一个开放的 assistant 后缀。这种简洁让 `tests/test_prompt.py` 可以直接用字符串比较验证渲染。

---

## 3. 为什么 Llama 被拒绝

Llama-3.2-1B 是一个**基础完成模型（base completion model）**。它在纯文本上训练，没有任何聊天模板——`tokenizer_config.json` 里没有 `chat_template` 字段，也没有任何角色边界特殊 token，更没有 instruction fine-tuning。

`format_chat_prompt(messages, "llama")` 直接抛出 `ValueError`，而不是悄悄编一份模板出来：

```python
# tiny_duo_infer/prompt.py
if model_type == "llama":
    raise ValueError(
        "Llama-3.2-1B is a base model without a chat template. "
        "Use a plain prompt string (chat=False) instead."
    )
```

这是有意为之的失败模式。如果 Llama 静默接受模板，会发生两件事：

1. 模型从未见过这种格式，会生成一种"看起来像噪声但其实是协议错配"的低质量输出。
2. 用 `--chat` 跑 Llama 的 CLI 用户得到看似合理却错误的补全，把 bug 误以为正常工作，然后让它进入下游评测。

模板阶段大声报错，远比推理阶段静默错误安全。如果某天需要一个 chat-tuned Llama 变体，应当改 `model_type` 并加入真正的模板，而不是粉饰当前的"无模板"事实。

---

## 4. 校验：ChatMessage 与 GenerationRequest

模板被调用之前，校验在两处提前发生：

### `ChatMessage.__post_init__`

```python
# tiny_duo_infer/generation.py
_VALID_ROLES = frozenset({"system", "user", "assistant"})

@dataclass
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(...)
        if not self.content:
            raise ValueError("ChatMessage content must not be empty.")
```

- `{system, user, assistant}` 之外的 role 一律被拒。tool / function-calling 角色超出 Phase 1.6 范围。
- 空 content 被拒。空 content 在 ChatML 下能产出语法上"合法"的 prompt，但其实是在请求模型注意一个空回合，这几乎总是调用方的 bug。

### `GenerationRequest.__post_init__`

```python
# tiny_duo_infer/generation.py
if self.prompt is None and self.messages is None:
    raise ValueError("Exactly one of 'prompt' or 'messages' must be provided.")
if self.prompt is not None and self.messages is not None:
    raise ValueError(...)
if self.messages is not None and not self.chat:
    raise ValueError("'messages' requires chat=True.")
if self.messages is not None and not self.messages:
    raise ValueError("'messages' must not be an empty list.")
```

`messages requires chat=True` 这条规则规避一个微妙的混淆：如果允许 `chat=False` 配合 messages 同时存在，可能会有人去把 `ChatMessage.content` 简单串起来当 plain prompt 使用。强制 `chat=True` 让"模板化这一步"在请求形状里变得显式。

---

## 5. 三个用户可见的入口面

同一份模板存在于三个入口背后；只是 wire 层格式不同。

### CLI：`--message ROLE:CONTENT`

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --message system:You are a helpful assistant. \
  --message user:What is 2+2?
```

`tiny_duo_infer/cli.py` 把每个 `--message` 参数按**首个**冒号切分成 `ChatMessage`——也就是说 content 中可以包含冒号（例如 `url:https://...`）。重复 `--message` 形成有序列表。

也存在一个简化形式：

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "What is 2+2?" \
  --chat
```

`--chat` 配合 `--prompt` 把 prompt 包成一条 `user` 消息。这个包装是在引擎里完成的，而不是 CLI——所以 HTTP 调用方发送 `chat: true` + 一个 plain `prompt` 时也享受同样行为。

### HTTP：`messages: [{role, content}]`

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Hi"}
    ],
    "chat": true,
    "max_new_tokens": 64
  }'
```

`tiny_duo_infer/serving/api.py:_to_generation_request` 把每个 pydantic `_ChatMessageBody` 转成 `ChatMessage`，于是引擎在两条代码路径上看到同一份 dataclass。`ChatMessage.__post_init__` 在转换时就跑——所以 HTTP 422（校验错误）会在 worker 介入之前返回。

### Python API：`GenerationRequest(messages=..., chat=True)`

最底层的入口是直接调 `Engine.generate_request()`。CLI 与 HTTP 都走它——任何用 Python 写的聊天模板测试都同时覆盖上层两个入口。

---

## 6. `format_chat_prompt` 的输出形态

对于 `messages = [system: "Be concise.", user: "Hi"]`、`model_type = "qwen3"`，渲染出的 prompt **完全等于**：

```text
<|im_start|>system
Be concise.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
```

把这串字符串传给 `Tokenizer.encode(prompt_str, add_special_tokens=True)`：

- `<|im_start|>` 与 `<|im_end|>` 解析为各自已注册的 special token ID，每个占用一个位置。
- 角色名（`system`、`user`、`assistant`）是普通词表文本，按 BPE merge 切成 1–2 个 token。
- 换行被编码为 `\n` 的 BPE 字节 token。
- `<|im_start|>assistant` 之后的换行**必须保留**——它是 prompt 的一部分，让模型生成的第一个 token 落在正确的字节边界上。

tokenization 之后引擎对所有这些做 prefill，然后采样 assistant 的第一个 token。因为 prompt 以一个开放的 assistant 回合结束，模型直接生成 assistant 内容——没有角色标记、没有多余空白、没有任何脚手架。

---

## 7. 阅读代码时要写下的清单

针对聊天模板这个面，写下：

- **输入：** `messages: list[ChatMessage]`、`model_type: str`。
- **输出：** 一个 `str`，可直接传给 `Tokenizer.encode(..., add_special_tokens=True)`。
- **张量形状：** 无——这是一个 tokenization 之前的格式化器。
- **状态：** 无——`format_chat_prompt` 是纯函数、确定性。
- **不变量：**
  - 输出对 Qwen3 永远以 `<|im_start|>assistant\n` 结尾。
  - 输出永远不以 `<|im_end|>\n` 结尾（assistant 回合保持开放）。
  - 输出中的角色顺序与输入列表完全一致。
  - Llama 永远抛错；Qwen3 在非空、合法 `ChatMessage` 列表上永远成功。
- **失败情况：** role 非法、content 为空、`messages` 列表为空、`model_type` 不被支持、Llama 配 `chat=True`。
- **静默生成 corruption 的危险点：** 把末尾的 `<|im_start|>assistant\n` 改成 `<|im_start|>assistant\n<|im_end|>\n` 仍然能 tokenize、仍然能 prefill。但模型会看到一个**已闭合**的 assistant 回合，并尝试开启**新角色**——结果是生成 `<|im_start|>user\n` 之类的内容，而不是答案。`tests/test_prompt.py` 中的 open-assistant 后缀断言就是为了防止这类回归。

---

## 延伸阅读

- `docs/phases/phase-1.6-generation-serving.md`：phase 权威 spec，含 chat 模板契约与 out-of-scope 清单。
- `tiny_duo_infer/prompt.py`：`format_chat_prompt`、`_format_chatml`。
- `tiny_duo_infer/generation.py`：`ChatMessage`、`GenerationRequest` 的校验规则。
- `tiny_duo_infer/cli.py`：`--message ROLE:CONTENT`、`--chat`。
- `tiny_duo_infer/serving/api.py`：pydantic `_ChatMessageBody`、`_to_generation_request`。
- `tests/test_prompt.py`：精确 ChatML 字符串断言，含 open-assistant 后缀锁定测试。
- `learning_materials/roadmap.md`：引导阅读顺序。
- `learning_materials/deep_dives/inference_worker.md`：把模板化后的 prompt 实际送进引擎跑的 worker 线程。
