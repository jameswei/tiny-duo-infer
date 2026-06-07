# Deep Dive: Chat Templating and the ChatML Protocol

This document is a focused walkthrough of the **chat prompt protocol** added in
Phase 1.6 of `tiny-duo-infer`. It pairs with the source files
`tiny_duo_infer/prompt.py` and `tiny_duo_infer/generation.py`, the unit tests in
`tests/test_prompt.py`, and the CLI/HTTP entry points in
`tiny_duo_infer/cli.py` and `tiny_duo_infer/serving/api.py`.

The Phase 1.6 goal is not a general-purpose templating engine. It is to make
visible the *contract* between the inference engine and a chat-style caller:
which special tokens are required, how role boundaries are encoded, why the
final assistant turn is left open, and which models can actually use chat mode.

---

## 1. Why The Engine Owns The Template

A trained chat model expects its prompt to follow the exact byte-level format
that was used during fine-tuning. Two boundaries must be reproduced precisely:

- **Role markers** — special tokens that the tokenizer treats as a single,
  registered unit. If the caller sends literal text like `"system: ..."`, the
  tokenizer will encode it as raw characters, which the model has never seen as
  a role boundary during training.
- **Turn separators** — newlines and an end-of-turn marker that signal where
  one role's content stops and the next role's content begins.

If the caller is responsible for assembling these boundaries by hand, every
client must learn the model's training-time format. Worse, mistakes are
silent — the model still generates *something*, just from a corrupted prompt.

Phase 1.6 fixes the contract on the engine side:

1. The HTTP/CLI layer accepts a structured `messages: list[ChatMessage]` with
   `role` and `content` fields.
2. The engine dispatches by `config.model_type` to the model's templating
   function in `tiny_duo_infer/prompt.py:format_chat_prompt`.
3. The output is a single tokenizable string ready for
   `Tokenizer.encode(prompt_str, add_special_tokens=True)`.

The caller never sees the special tokens. The engine never sees a malformed
prompt.

---

## 2. The ChatML Format (Qwen3)

Qwen3 uses the **ChatML** format from its `tokenizer_config.json`:

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

Two strict rules govern this format:

| Rule | Why |
|---|---|
| `<|im_start|>` and `<|im_end|>` are single tokens, registered as special in `tokenizer.json`. | Without `AddedToken(special=True)` registration the tokenizer would split them into byte fragments, which Qwen3 has never seen. The tokenizer wrapper section of the roadmap (§3) covers this lookup path. |
| The final `<|im_start|>assistant\n` is **not** closed with `<|im_end|>`. | This is the *generation prompt*. By leaving the assistant turn open, the model is told "your job is to continue from here." Closing it with `<|im_end|>` would tell the model the turn already ended and it should produce a new role marker next, not assistant content. |

### Tiny-duo-infer implementation

```python
# tiny_duo_infer/prompt.py
def _format_chatml(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{_IM_START}{msg.role}\n{msg.content}{_IM_END}\n")
    parts.append(f"{_IM_START}assistant\n")
    return "".join(parts)
```

There is no branching on prior assistant content, no stripping of trailing
whitespace, no special handling of system messages. The format is the loop
plus the open assistant suffix. That keeps the rendering verifiable by string
comparison in `tests/test_prompt.py`.

---

## 3. Why Llama Is Rejected

Llama-3.2-1B is a **base completion model**. It was trained on plain text and
has no associated chat template — there is no `chat_template` field in its
`tokenizer_config.json`, no role-boundary special tokens, no instruction
fine-tuning.

`format_chat_prompt(messages, "llama")` raises `ValueError` rather than
silently fabricating a template:

```python
# tiny_duo_infer/prompt.py
if model_type == "llama":
    raise ValueError(
        "Llama-3.2-1B is a base model without a chat template. "
        "Use a plain prompt string (chat=False) instead."
    )
```

This is a deliberate failure mode. Two things would happen if Llama silently
accepted a template:

1. The model would generate from a prompt it has never seen during training,
   producing low-quality output that *looks* like normal generation noise but
   is actually a protocol mismatch.
2. CLI users running `--chat` against Llama would get plausible-but-wrong
   completions and assume the engine works, which propagates the bug into
   downstream evaluations.

A loud error at template time is safer than a silent error at inference time.
Anything that needs a chat-tuned Llama variant should change `model_type` and
add a real template, not paper over the absence of one.

---

## 4. Validation: ChatMessage And GenerationRequest

Validation runs in two places before the template is even reached:

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

- Roles outside `{system, user, assistant}` are rejected. Tools/function-calling
  roles are out of scope for Phase 1.6.
- Empty content is rejected. The ChatML format produces a syntactically valid
  prompt for empty content, but that prompt asks the model to attend to a turn
  with no information, which is almost always a caller bug.

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

The `messages requires chat=True` rule prevents a subtle confusion: it would
be possible to silently concatenate `ChatMessage.content` strings into a plain
prompt if `chat=False` were allowed with messages. Forcing `chat=True` makes
the templating step explicit in the request shape.

---

## 5. The Three User-Facing Surfaces

The same template lives behind all three entry points; only the wire format
differs.

### CLI: `--message ROLE:CONTENT`

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --message system:You are a helpful assistant. \
  --message user:What is 2+2?
```

`tiny_duo_infer/cli.py` parses each `--message` argument into a `ChatMessage`
by splitting on the **first** colon only — content can therefore contain
colons (`url:https://...`). Repeating `--message` builds the ordered list.

A simpler shorthand also exists:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "What is 2+2?" \
  --chat
```

`--chat` with `--prompt` wraps the prompt as a single `user` message. This is
implemented inside the engine, not the CLI, so the same shorthand works for
HTTP callers that send `chat: true` with a plain `prompt`.

### HTTP: `messages: [{role, content}]`

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

`tiny_duo_infer/serving/api.py:_to_generation_request` converts each pydantic
`_ChatMessageBody` to a `ChatMessage` so the engine sees the same dataclass
on both code paths. `ChatMessage.__post_init__` runs at conversion time, so
HTTP 422 (validation error) is returned before any worker is involved.

### Python API: `GenerationRequest(messages=..., chat=True)`

The lowest-level surface is `Engine.generate_request()` directly. CLI and HTTP
both go through this function, so any chat-template behavior tested via Python
also covers the two upper layers.

---

## 6. Shape Of `format_chat_prompt`'s Output

For `messages = [system: "Be concise.", user: "Hi"]` with `model_type = "qwen3"`,
the rendered prompt is exactly:

```text
<|im_start|>system
Be concise.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
```

When this string is passed to `Tokenizer.encode(prompt_str, add_special_tokens=True)`:

- `<|im_start|>` and `<|im_end|>` resolve to their registered special-token
  IDs and consume a single position each.
- The role names (`system`, `user`, `assistant`) are normal vocabulary text,
  one or two tokens depending on the BPE merges.
- Newlines are encoded as the BPE byte token for `\n`.
- The trailing newline after `<|im_start|>assistant` is part of the prompt and
  must be present so the model's first generated token starts at the right
  byte boundary.

After tokenization the engine runs `prefill` over all of this, then samples
the assistant's first token. Because the prompt ends in an open assistant
turn, the model produces assistant content directly — no role marker, no
extra whitespace, no scaffolding.

---

## 7. What To Verify When You Read The Code

For the chat-template surface, write down:

- **Inputs:** `messages: list[ChatMessage]`, `model_type: str`.
- **Outputs:** a single `str`, ready for tokenization with
  `add_special_tokens=True`.
- **Tensor shapes:** none — this is a pre-tokenization formatter.
- **State:** none — `format_chat_prompt` is pure and deterministic.
- **Invariants:**
  - The output always ends with `<|im_start|>assistant\n` for Qwen3.
  - The output never ends with `<|im_end|>\n` (the assistant turn is open).
  - Role order in the output matches the input list order exactly.
  - Llama always raises; Qwen3 always succeeds for a non-empty list of valid
    `ChatMessage`s.
- **Failure cases:** invalid role, empty content, empty `messages` list,
  unsupported `model_type`, Llama with `chat=True`.
- **The one thing that would silently corrupt generation:** changing the
  trailing `<|im_start|>assistant\n` into `<|im_start|>assistant\n<|im_end|>\n`
  would still tokenize and still run prefill, but the model would see a
  closed assistant turn and try to start a *new* role next, producing
  `<|im_start|>user\n` content instead of an answer. The unit tests in
  `tests/test_prompt.py` lock the open-assistant suffix specifically to catch
  this regression.

---

## Further Reading

- `docs/phases/phase-1.6-generation-serving.md` — authoritative phase spec
  including the chat-template contract and out-of-scope items.
- `tiny_duo_infer/prompt.py` — `format_chat_prompt`, `_format_chatml`.
- `tiny_duo_infer/generation.py` — `ChatMessage`, `GenerationRequest`
  validation rules.
- `tiny_duo_infer/cli.py` — `--message ROLE:CONTENT`, `--chat`.
- `tiny_duo_infer/serving/api.py` — pydantic `_ChatMessageBody`,
  `_to_generation_request`.
- `tests/test_prompt.py` — exact ChatML string assertions, including the
  open-assistant-suffix lock.
- `learning_materials/roadmap.md` — guided reading order.
- `learning_materials/deep_dives/inference_worker.md` — the worker thread
  that actually runs the templated prompt through the engine.
