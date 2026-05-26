# 数据流图

展示一次生成请求中每个阶段的张量形状。配合 [内部原理](internals_zh.md) 和
[Phase 规范](../docs/phases/phase-1-mlx-single-user.md#tensor-shape-conventions) 一起阅读。

---

## Prefill（预填充）

```
                           ┌───────────────────────────────────────┐
                           │           Engine.generate()           │
                           └───────────────────┬───────────────────┘
                                               │
                                               ▼
                           ┌───────────────────────────────────────┐
                           │           Engine.prefill()            │
                           └───────────────────┬───────────────────┘
                                               │
                                     prompt: "The capital of..."
                                               │
                                               ▼
┌─────────────────┐              ┌──────────────────────────────────┐
│   tokenizer.json │────────────▶│   Tokenizer.encode(prompt)       │
│   tokenizer_config.json       │   → [128000, 791, 14013, ...]     │
└─────────────────┘              └───────────────────┬──────────────┘
                                                    │
                                          token_ids: list[int]
                                          prompt_len = S
                                                    │
                         ┌──────────────────────────┼──────────────────────────┐
                         │                          ▼                          │
                         │              ┌───────────────────────┐              │
                         │              │   Engine._new_cache() │              │
                         │              │   KVCache(n_layers,   │              │
                         │              │     n_kv_heads,       │              │
                         │              │     max_seq_len,      │              │
                         │              │     head_dim)         │              │
                         │              └───────────┬───────────┘              │
                         │                          │                          │
                         │              K: (1,8,T_max,64) × 16 *               │
                         │              V: (1,8,T_max,64) × 16 *               │
                         │              current_len = 0                        │
                         │                          │                          │
                         │              ┌───────────▼───────────┐              │
                         ▼              │                        │              ▼
               input_ids: (1,S)    cache (empty)        position_offset=0
                         │              │                        │
                         │    ┌─────────┴────────┐               │
                         │    │                  │               │
                         ▼    ▼                  ▼               ▼
               ┌─────────────────────────────────────────────────────┐
               │              LlamaModel.forward()                   │
               │                                                     │
               │  ┌──────────────────────────────────────────────┐  │
               │  │  embed_tokens(input_ids) → (1, S, 2048)      │  │
               │  └────────────────────┬─────────────────────────┘  │
               │                      │                              │
               │                      ▼                              │
               │  ┌──────────────────────────────────────────────┐  │
               │  │  LlamaBlock × 16                             │  │
               │  │                                              │  │
               │  │  For each block i:                           │  │
               │  │    1. input_norm(x)  → (1, S, 2048)         │  │
               │  │    2. attn(normed_x) → (1, S, 2048)         │  │
               │  │       ├─ q_proj → (1, S, 32, 64)            │  │
               │  │       ├─ k_proj → (1, S, 8, 64)             │  │
               │  │       ├─ v_proj → (1, S, 8, 64)             │  │
               │  │       ├─ apply_rope(q, offset=0)            │  │
               │  │       ├─ apply_rope(k, offset=0)            │  │
               │  │       ├─ cache.update(i, k, v, pos=0)       │  │
               │  │       │    → k_full (1,8,S,64)              │  │
               │  │       │    → v_full (1,8,S,64)              │  │
               │  │       ├─ repeat K/V → (1,32,S,64)           │  │
               │  │       ├─ Q·K^T/√64 → (1,32,S,S)            │  │
               │  │       ├─ causal_mask → (1,32,S,S)           │  │
               │  │       ├─ softmax → (1,32,S,S)               │  │
               │  │       ├─ weighted_sum → (1,32,S,64)         │  │
               │  │       └─ o_proj → (1, S, 2048)              │  │
               │  │    3. residual: x = x + attn_out             │  │
               │  │    4. post_attn_norm(x) → (1, S, 2048)      │  │
               │  │    5. ffn(normed_x) → (1, S, 2048)          │  │
               │  │       ├─ gate_proj → (1, S, 8192)           │  │
               │  │       ├─ up_proj   → (1, S, 8192)           │  │
               │  │       ├─ silu(gate) * up → (1, S, 8192)    │  │
               │  │       └─ down_proj → (1, S, 2048)           │  │
               │  │    6. residual: x = x + ffn_out              │  │
               │  └────────────────────┬─────────────────────────┘  │
               │                      │                              │
               │                      ▼                              │
               │  ┌──────────────────────────────────────────────┐  │
               │  │  final_norm(x) → (1, S, 2048)               │  │
               │  └────────────────────┬─────────────────────────┘  │
               │                      │                              │
               │                      ▼                              │
               │  ┌──────────────────────────────────────────────┐  │
               │  │  lm_head(x) → (1, S, 128256)                │  │
               │  └────────────────────┬─────────────────────────┘  │
               └───────────────────────┼────────────────────────────┘
                                       │
                             logits: (1, S, 128256)
                                       │
                         ┌─────────────▼─────────────┐
                         │  logits[0, S-1, :]        │
                         │  → final_logits (128256,) │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │  cache.advance(S)         │  ◄── 提交全部 16 层的写入
                         │  current_len = S          │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │  mx.eval(final_logits)    │  ◄── 为 CPU 采样具体化
                         │  cache.eval()             │      为 decode 读取具体化
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │  sample(final_logits)     │
                         │  → token "Paris" (3663)   │
                         └─────────────┬─────────────┘
                                       │
                              ┌────────┴────────┐
                              │  进入 decode 循环  │
                              └───────────────────┘
```

### Mermaid — Prefill

```mermaid
flowchart TD
    A["Engine.generate(prompt)"] --> B["Engine.prefill()"]
    B --> C["Tokenizer.encode(prompt)\n→ [128000, 791, 14013, ...]\nprompt_len = S"]

    C --> D["Engine._new_cache()\nK: (1,8,T_max,64) × 16\nV: (1,8,T_max,64) × 16\ncurrent_len = 0"]
    C --> E["input_ids: (1, S)"]
    C --> F["position_offset = 0"]

    D --> G["LlamaModel.forward()"]
    E --> G
    F --> G

    subgraph G["LlamaModel.forward()"]
        direction TB
        G1["embed_tokens\n(1,S) → (1,S,2048)"] --> G2["LlamaBlock × 16"]
        subgraph G2["LlamaBlock (per block i)"]
            direction TB
            H1["input_norm → (1,S,2048)"] --> H2["LlamaAttention"]
            subgraph H2["LlamaAttention"]
                direction LR
                A1["q_proj\n(1,S,32,64)"] --> A2["apply_rope(q, offset=0)"]
                A3["k_proj\n(1,S,8,64)"] --> A4["apply_rope(k, offset=0)"]
                A5["v_proj\n(1,S,8,64)"] --> A6["cache.update(i, k, v, pos=0)\n→ k_full (1,8,S,64)\n→ v_full (1,8,S,64)"]
                A2 --> A7["repeat K/V → (1,32,S,64)"]
                A4 --> A6
                A6 --> A7
                A7 --> A8["Q·Kᵀ/√64\n(1,32,S,S)"]
                A8 --> A9["causal_mask"]
                A9 --> A10["softmax\n(1,32,S,S)"]
                A10 --> A11["weighted_sum\n(1,32,S,64)"]
                A11 --> A12["o_proj\n(1,S,2048)"]
            end
            H2 --> H3["residual: x + attn_out"]
            H3 --> H4["post_attn_norm → (1,S,2048)"]
            H4 --> H5["SwiGLUFFN"]
            subgraph H5["SwiGLUFFN"]
                direction LR
                F1["gate_proj\n(1,S,8192)"] --> F2["silu(gate) * up"]
                F3["up_proj\n(1,S,8192)"] --> F2
                F2 --> F4["down_proj\n(1,S,2048)"]
            end
            H5 --> H6["residual: x + ffn_out"]
        end
        G2 --> G3["final_norm\n(1,S,2048)"]
        G3 --> G4["lm_head\n(1,S,128256)"]
    end

    G --> I["logits[0, S-1, :]\n→ final_logits (128256,)"]
    I --> J["cache.advance(S)\ncurrent_len = S"]
    J --> K["mx.eval(final_logits)\ncache.eval()\n◄ 具体化 logits 用于采样\n和缓存缓冲区用于 decode"]
    K --> L["sample(final_logits)\n→ 第一个生成的 token"]

    style A fill:#e1f5fe
    style L fill:#e8f5e9
    style J fill:#fff3e0
    style K fill:#fff3e0
```

---

## Decode Loop（解码循环）

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Engine.generate()  decode loop                    │
│                                                                     │
│  current_len = S (来自 prefill)                                      │
│  next_token  = 第一个采样的 token（来自 prefill logits）              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  for step in range(max_new_tokens):                           │  │
│  │                                                               │  │
│  │    ┌─ EOS? ──────────────────────────────────────────────┐   │  │
│  │    │   yes → break（不 yield EOS token）                  │   │  │
│  │    └──────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │    ┌─ yield tokenizer.decode([next_token]) ─────────────┐     │  │
│  │    │   例如 " Paris", ",", " located", " in", " the"    │     │  │
│  │    └──────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │    ┌─ 最后一步？ ────────────────────────────────────────┐     │  │
│  │    │   yes → break（不浪费最后一次 forward）              │     │  │
│  │    └──────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │    ┌─────────────────────────────────────────────────────┐    │  │
│  │    │                DECODE FORWARD                       │    │  │
│  │    │                                                     │    │  │
│  │    │  input_ids = mx.array([[next_token]])               │    │  │
│  │    │            = (1, 1)                                 │    │  │
│  │    │  position_offset = cache.current_len                │    │  │
│  │    │                                                     │    │  │
│  │    │  ┌─────────────────────────────────────────────┐   │    │  │
│  │    │  │  embed_tokens(input_ids) → (1, 1, 2048)    │   │    │  │
│  │    │  └──────────────────┬──────────────────────────┘   │    │  │
│  │    │                     │                              │    │  │
│  │    │                     ▼                              │    │  │
│  │    │  ┌─────────────────────────────────────────────┐   │    │  │
│  │    │  │  LlamaBlock × 16 (decode 模式, S=1)        │   │    │  │
│  │    │  │                                             │   │    │  │
│  │    │  │  For each block i:                          │   │    │  │
│  │    │  │    input_norm(x) → (1, 1, 2048)            │   │    │  │
│  │    │  │                                             │   │    │  │
│  │    │  │    LlamaAttention (decode):                 │   │    │  │
│  │    │  │      q_proj → (1, 1, 32, 64)               │   │    │  │
│  │    │  │      k_proj → (1, 1, 8, 64)                │   │    │  │
│  │    │  │      v_proj → (1, 1, 8, 64)                │   │    │  │
│  │    │  │      apply_rope(q, offset=cache.current_len)│   │    │  │
│  │    │  │      apply_rope(k, offset=cache.current_len)│   │    │  │
│  │    │  │      cache.update(i, k, v, pos=current_len) │   │    │  │
│  │    │  │        → k_full (1, 8, T, 64)   T=S+step   │   │    │  │
│  │    │  │        → v_full (1, 8, T, 64)             │   │    │  │
│  │    │  │      repeat K/V → (1, 32, T, 64)           │   │    │  │
│  │    │  │      Q·K^T/√64 → (1, 32, 1, T)            │   │    │  │
│  │    │  │      mask → 空操作（所有 key 都在过去）     │   │    │  │
│  │    │  │      softmax → (1, 32, 1, T)               │   │    │  │
│  │    │  │      weighted_sum → (1, 32, 1, 64)         │   │    │  │
│  │    │  │      o_proj → (1, 1, 2048)                 │   │    │  │
│  │    │  │                                             │   │    │  │
│  │    │  │    residual: x + attn_out                    │   │    │  │
│  │    │  │    post_attn_norm → SwiGLUFFN → residual    │   │    │  │
│  │    │  └──────────────────┬──────────────────────────┘   │    │  │
│  │    │                     │                              │    │  │
│  │    │                     ▼                              │    │  │
│  │    │    final_norm(x) → (1, 1, 2048)                   │    │  │
│  │    │    lm_head(x)    → (1, 1, 128256)                 │    │  │
│  │    └─────────────────────┬──────────────────────────────┘    │  │
│  │                          │                                   │  │
│  │                          ▼                                   │  │
│  │    ┌─────────────────────────────────────────────────────┐   │  │
│  │    │  mx.eval(logits)                                    │   │  │
│  │    │  cache.eval()                                       │   │  │
│  │    │  cache.advance(1)          current_len += 1         │   │  │
│  │    └───────────────────────────┬─────────────────────────┘   │  │
│  │                                │                             │  │
│  │    ┌───────────────────────────▼─────────────────────────┐   │  │
│  │    │  logits[0, 0, :] → (128256,)                       │   │  │
│  │    │  sample(...) → next_token                          │   │  │
│  │    └─────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │    ◄──────── 用新的 next_token 回到循环开头 ─────────────────┘  │
│  └──────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

### Mermaid — Decode Loop

```mermaid
flowchart TD
    A["next_token\n（来自 prefill 采样）"] --> B{"EOS?"}
    B -->|yes| C["break\n（不 yield EOS）"]
    B -->|no| D["yield\ntokenizer.decode([next_token])"]
    D --> E{"最后一步\n(max_new_tokens-1)?"}
    E -->|yes| F["break\n（不浪费 forward）"]
    E -->|no| G["DECODE FORWARD"]
    F --> H["done"]
    C --> H

    subgraph G["Decode Forward"]
        direction TB
        G0["input_ids = mx.array([next_token])\n= (1, 1)\nposition_offset = cache.current_len"] --> G1
        G1["embed_tokens\n(1,1) → (1,1,2048)"] --> G2["LlamaBlock × 16"]

        subgraph G2["LlamaBlock (decode, S=1)"]
            direction TB
            D1["input_norm → (1,1,2048)"] --> D2["LlamaAttention (decode)"]
            subgraph D2["LlamaAttention"]
                direction LR
                Q1["q_proj\n(1,1,32,64)"] --> Q2["apply_rope(q)\noffset=cache.current_len"]
                K1["k_proj\n(1,1,8,64)"] --> K2["apply_rope(k)\noffset=cache.current_len"]
                V1["v_proj\n(1,1,8,64)"] --> V2["cache.update(i, k, v, pos=current_len)\n→ k_full (1,8,T,64)\n→ v_full (1,8,T,64)"]
                K2 --> V2
                Q2 --> A1["repeat K/V → (1,32,T,64)"]
                V2 --> A1
                A1 --> A2["Q·Kᵀ/√64\n(1,32,1,T)"]
                A2 --> A3["mask → 空操作\n（所有 key 都在过去）"]
                A3 --> A4["softmax\n(1,32,1,T)"]
                A4 --> A5["weighted_sum\n(1,32,1,64)"]
                A5 --> A6["o_proj\n(1,1,2048)"]
            end
            D2 --> D3["residual: x + attn_out"]
            D3 --> D4["post_attn_norm → SwiGLUFFN → residual"]
        end
        G2 --> G3["final_norm\n(1,1,2048)"]
        G3 --> G4["lm_head\n(1,1,128256)"]
    end

    G --> I["mx.eval(logits)\ncache.eval()\ncache.advance(1)\ncurrent_len += 1"]
    I --> J["logits[0,0,:]\n→ (128256,)"]
    J --> K["sample(...)\n→ next_token"]
    K --> B

    style A fill:#e8f5e9
    style C fill:#ffcdd2
    style F fill:#ffcdd2
    style H fill:#eeeeee
    style I fill:#fff3e0
    style K fill:#e8f5e9
```

---

## 完整流程 —— 端到端

```
┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────────────┐
│  config   │    │ tokenizer │    │  safetensors  │    │    用户 prompt    │
│  .json    │    │  .json    │    │  (HF 权重)    │    │                  │
└─────┬─────┘    └─────┬─────┘    └───────┬──────┘    └────────┬─────────┘
      │                │                  │                     │
      ▼                ▼                  ▼                     │
┌───────────┐   ┌───────────┐   ┌───────────────┐               │
│ModelConfig│   │ Tokenizer │   │ load_weights() │               │
│           │   │ .encode() │   │ convert()      │               │
│           │   │ .decode() │   │ load_weights() │               │
└─────┬─────┘   └─────┬─────┘   └───────┬───────┘               │
      │               │                 │                        │
      │    ┌──────────┴─────────┐       │                        │
      │    │                    │       │                        │
      ▼    ▼                    ▼       ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Engine.from_model_path()                    │
│                                                                     │
│   config = load_config(model_path)                                  │
│   tokenizer = Tokenizer.from_pretrained(model_path)                 │
│   hf_weights = load_weights(model_path)                             │
│   project_weights = convert(hf_weights, config)                     │
│   model = LlamaModel(config)                                        │
│   model.load_weights(project_weights)                               │
│                                                                     │
│   → Engine(model, tokenizer, config, max_seq_len)                   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Engine.generate(prompt)                        │
│                                                                     │
│  ┌──────── PREFILL ────────┐                                        │
│  │ tokenize → forward(S,0) │                                        │
│  │ → 填充 cache[0:S]       │                                        │
│  │ → eval logits + cache   │                                        │
│  │ → 采样第 1 个 token      │                                        │
│  └──────────┬──────────────┘                                        │
│             │                                                       │
│             ▼                                                       │
│  ┌──────── DECODE LOOP ────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  对每个 token:                                               │    │
│  │    yield decode(token_text)                                 │    │
│  │    forward(next_token, current_len)                         │    │
│  │    → 写入 cache[current_len]                                │    │
│  │    → eval logits + cache                                    │    │
│  │    → advance(1)                                             │    │
│  │    → 采样下一个 token                                        │    │
│  │                                                             │    │
│  │  停止: EOS 或 max_new_tokens                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                         生成的文本流
```

### Mermaid — 完整流程

```mermaid
flowchart TD
    subgraph INIT["Engine.from_model_path()"]
        direction TB
        CFG["config.json\n→ ModelConfig"] --> ENG
        TOK["tokenizer.json\n→ Tokenizer\n.encode() .decode()"] --> ENG
        WGT["safetensors\n→ load_weights()\n→ convert()"] --> ENG
        M["LlamaModel(config)\nmodel.load_weights()"] --> ENG
        ENG["Engine(model, tokenizer, config, max_seq_len)"]
    end

    INIT --> GEN

    subgraph GEN["Engine.generate(prompt)"]
        direction TB

        subgraph PF["PREFILL（1 次前向传播）"]
            direction TB
            P1["tokenize prompt → token_ids"] --> P2["KVCache(max_seq_len)\ncurrent_len = 0"]
            P1 --> P3["model(input_ids=(1,S), position_offset=0)\n→ 填充 cache[0:S]"]
            P2 --> P3
            P3 --> P4["cache.advance(S)\ncurrent_len = S\nmx.eval(final_logits)\ncache.eval()"]
            P4 --> P5["sample → 第一个 token"]
        end

        PF --> D0

        subgraph DL["DECODE LOOP"]
            direction TB
            D0{"EOS?"} -->|yes| DONE["停止"]
            D0 -->|no| D1["yield decode(token)"]
            D1 --> D2{"最后一步？"}
            D2 -->|yes| DONE
            D2 -->|no| D3["model(input_ids=(1,1),\nposition_offset=current_len)\n→ 写入 cache[current_len]"]
            D3 --> D4["mx.eval(logits)\ncache.eval()\ncache.advance(1)\ncurrent_len += 1"]
            D4 --> D5["sample → 下一个 token"]
            D5 --> D0
        end

        DONE --> OUT["生成的文本流"]
    end

    style INIT fill:#e3f2fd
    style PF fill:#e8f5e9
    style DL fill:#fff8e1
    style P4 fill:#fff3e0
    style D4 fill:#fff3e0
    style OUT fill:#f3e5f5
```

---

## 关键形状速查

| 符号 | 含义 | Llama-3.2-1B |
|--------|---------|--------------|
| `B` | batch 大小 | 1 |
| `S` | 序列长度（prefill: prompt_len; decode: 1） | 可变 |
| `T` | KV 缓存总长度（每步增长） | 可变 |
| `D` | 隐藏维度 | 2048 |
| `H` | 查询头数 | 32 |
| `Hkv` | 键/值头数（GQA） | 8 |
| `Dh` | 头维度 | 64 |
| `V` | 词表大小 | 128256 |
| `L` | Transformer 层数 | 16 |
| `I` | FFN 中间维度 | 8192 |
| `n_groups` | GQA 分组数（H / Hkv） | 4 |

## KV 缓存内存

```
每层、每个缓冲区:  (1, Hkv, T, Dh) × dtype_bytes
总计:              2 × L × Hkv × T × Dh × 2 字节 (bfloat16)

T=1024:  2 × 16 × 8 × 1024 × 64 × 2 = 33,554,432 字节 ≈ 32 MB
T=2048:  2 × 16 × 8 × 2048 × 64 × 2 = 67,108,864 字节 ≈ 64 MB
```
