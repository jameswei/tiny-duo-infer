# 推理引擎内部原理

深入解析 Phase 1 中最难理解的三个核心概念。建议在阅读
[学习路线图](roadmap.md) 之后或结合源代码一起阅读。

---

## 1. Prefill 与 Decode —— 推理引擎的双阶段设计

每次 LLM 的 token 生成都分为两个截然不同的阶段。理解*为什么*它们必须分开，
是构建推理引擎最重要的一课。

### Prefill：一次性处理整个 Prompt

当用户输入 "The capital of France is" 时，引擎看到的是：

```
prompt tokens:  [128000, 791, 14013, 315, 11405, 374]
                 BOS     The   cap    ital  of    France  is
```

全部 6 个 token 在**一次前向传播**中送入模型，`input_ids` 形状为 `(B=1, S=6)`。
模型依次计算：

1. **词嵌入（Embedding）：** `(1, 6, 2048)`
2. **16 个 Transformer 块** — 每个块*同时*处理所有位置 0..5
3. **最终归一化 + lm_head：** `(1, 6, 128256)` logits

引擎只保留 `logits[0, 5, :]`（最后一个位置）。这些 logits 预测第 7 个 token
—— 也就是第一个*生成*的 token。

**为什么一次完成？** GPU 可以并行处理所有 prompt token（一次矩阵乘法就能
覆盖全部 6 个位置）。这是快速的部分。

### Decode：逐个 Token 生成

第 7 个 token（"Paris"）成为*下一次*模型调用的输入：

```
decode step 1:  input_ids = [[3663]]           (B=1, S=1)  position_offset=6
decode step 2:  input_ids = [[<next_token>]]   (B=1, S=1)  position_offset=7
...
```

每一步都要跑完整的 16 层 Transformer，从 KV 缓存中读取所有历史位置，
然后生成一个新 token。这是慢速的部分 —— 它天生就是串行的。

### 为什么不批量 Decode？

你可能会想：为什么不把所有已生成的 token 像 prefill 一样批量送回去？
因为每个新 token 都依赖于前一个 token 的输出。引擎*必须*先采样出 token 7
才能计算 token 8 的嵌入。自回归生成是一条串行的依赖链 —— 没有办法绕过。

### Position Offset —— 连接两个阶段的桥梁

```python
# Prefill
model(input_ids, cache, position_offset=0)
# → RoPE 旋转位置 0, 1, 2, 3, 4, 5
# → causal mask: 位置 i 只能关注 [0..i]

# Decode 第 t 步（t = prompt_len + 已生成的 token 数）
model(input_ids, cache, position_offset=t)
# → RoPE 以绝对位置 t 旋转单个 token
# → causal mask: 可以关注缓存中全部 T 个位置（都在过去）
```

`position_offset` 确保 RoPE 编码和因果遮罩都基于完整序列中的*绝对*位置来计算，
而不是模型在 decode 时看到的那个小小的 `(B=1, S=1)` 切片。

### mx.eval() 的边界

MLX 采用惰性求值（lazy evaluation）：张量操作构建计算图，但直到调用
`mx.eval()` 才会真正执行。Phase 1 只在引擎边界处执行求值：

```
prefill forward  →  mx.eval(final_logits)  +  cache.eval()  →  sample
                                                                    ↓
decode forward   →  mx.eval(logits)        +  cache.eval()  →  sample
                                                                    ↓
decode forward   →  ...                                            ...
```

在 attention 或 FFN 层内部执行 eval 会：
- 增加不必要的 GPU/CPU 同步点
- 模糊推理步骤的起止边界
- 让惰性计算图更难理解

每次 forward 之后的两次 `eval()` 调用**不是**冗余的：

| 调用 | 具体化什么 | 为什么 |
|---|---|---|
| `mx.eval(logits)` | `(1, 1, V)` 输出 | CPU 端采样需要读取 token ID |
| `cache.eval()` | 全部 16 层的 K/V 缓冲区 | 下一个 decode 步骤需要读取这些缓冲区 |

它们服务于不同的消费者：`logits` 送给 CPU 采样器，缓存的 K/V 留在加速器上
给下一次 attention 使用。

---

## 2. KV 缓存的生命周期 —— `update()` 与 `advance()` 的分离

KV 缓存是引擎的长期记忆。它存储每个 token 的 key 和 value 向量，
这样模型就不需要重复计算它们。

### 缓冲区布局

```
每层:  K: (1, n_kv_heads=8, max_seq_len, head_dim=64)
       V: (1, n_kv_heads=8, max_seq_len, head_dim=64)

16 层 × 2 个缓冲区 × (1 × 8 × T × 64 × 2 字节) ≈ 32,768 × T 字节
T=1024 时: ~32 MB
```

`(..., max_seq_len, ...)` 维度是一次性预分配的。任何时刻只有
`[:, :, :current_len, :]` 切片是有效的。

### 两阶段写入协议

这里是大多数 KV 缓存 bug 的滋生地。Phase 1 有意将写入和推进分离：

```
update(layer, new_k, new_v, position)  ← 每个 token 步调用 16 次
advance(n_tokens)                      ← 每个 token 步只调用一次（由引擎调用）
```

**为什么不合并？** 在一个 decode 步骤中，全部 16 层都在位置 `p` 写入各自的
K/V。如果第一层就把 `current_len` 推进到 `p+1`，第二层就会在错误的位置上计算
RoPE，并且基于过时的缓存状态做 attention 遮罩。这种分离确保每一层在整次前向
传播中都看到相同的 `current_len`。

### Prefill 生命周期

```
第 0 步（prefill，prompt_len=6）:
  对全部 16 层:
    cache.update(layer, new_k, new_v, position=0)  → 返回 [:, :, :6, :]
  cache.advance(6)  → current_len 变成 6

  缓存状态: 位置 [0..5] 有效，current_len = 6
  Logits: (128256,) 来自位置 5
```

### Decode 生命周期

```
第 1 步（decode，token "Paris"）:
  position_offset = cache.current_len = 6  ← 引擎传入
  对全部 16 层:
    new_k, new_v: (1, Hkv, 1, Dh)  — 一个 token
    cache.update(layer, new_k, new_v, position=6)  → 返回 [:, :, :7, :]
  Forward 产生 logits (1, 1, V)
  mx.eval(logits)  +  cache.eval()
  cache.advance(1)  → current_len = 7
  从 logits[0, 0, :] 采样下一个 token

  缓存状态: 位置 [0..6] 有效，current_len = 7
```

### 为什么预分配而不是动态增长？

有些实现每步追加 K/V 张量（`torch.cat`）。概念上更简单，但每步都要复制整个
缓存缓冲区 —— 每个 token O(T)，总计 O(T²)。预分配避免了这个问题：写入只是
O(1) 的索引赋值。代价是必须提前知道 `max_seq_len`。

Phase 3 会用 **PagedAttention** 替代此方案：从共享池中分配固定大小的 KV 页，
同时消除复制问题和预分配浪费。

---

## 3. GQA + RoPE —— 为什么 Head 扩展很重要

### GQA（分组查询注意力）

Llama-3.2-1B 有 32 个查询头但只有 8 个键/值头。每个 KV 头被
`n_groups = 4` 个查询头共享：

```
Q heads:  [0, 1, 2, 3] → KV head 0
          [4, 5, 6, 7] → KV head 1
          ...以此类推
```

这将 KV 缓存内存减少到原来的 1/4，质量损失微乎其微。注意力计算时，
将 KV 头扩展回完整的 Q 头数量：

```python
k_expanded = mx.repeat(k_full, repeats=n_groups, axis=1)
# (1, 8, T, 64) → (1, 32, T, 64)
```

`axis=1` 的选择是刻意的。KV 头在转置后沿 head 维度（axis=1）存储。
在这个轴上重复意味着：
- `axis=0`（batch）：Phase 1 中已经是 1，多 batch 时会出错
- `axis=1`（heads）：按组排列查询头 —— Hkv 头 [0,0,0,0,1,1,1,1,...]
- `axis=2`（sequence）：会重复 token 而不是头

### RoPE（旋转位置编码）

RoPE 通过旋转每个头向量中连续的元素对来编码位置：

```
对于 head 维度中的每一对 (x0, x1):
  x0' = x0·cos(pos·θ_i) - x1·sin(pos·θ_i)
  x1' = x0·sin(pos·θ_i) + x1·cos(pos·θ_i)
```

频率 `θ_i` 在靠后的元素对中递减，形成一个从细粒度（快速旋转、对位置敏感）
到粗粒度（缓慢旋转、侧重语义）的编码谱系。

关键性质：
- **相对位置**：RoPE 之后 Q 和 K 的点积取决于 `pos_Q - pos_K`，
  自然地编码了相对距离
- **只应用于 Q 和 K**：V 不参与旋转，因为 attention 权重已经编码了位置关系；
  V 只存储内容
- **无学习参数**：旋转完全由位置决定

### 交互：RoPE 在扩展之前

RoPE 在 KV 缓存存储*之前*、在 GQA 头扩展*之前*应用于 Q 和 K：

```
1. Q 投影 → reshape → (B, S, H=32, Dh=64)
2. K 投影 → reshape → (B, S, Hkv=8, Dh=64)
3. apply_rope(q, ...)   ← 每个头独立旋转
4. apply_rope(k, ...)
5. k = transpose(k, ...) → (B, Hkv=8, S, Dh)     ← 准备写入缓存
6. cache.update(layer, k, v, position)
7. k_full, v_full = ... → (B, Hkv=8, T, Dh)     ← 缓存读回后
8. k_expanded = repeat(k_full, n_groups, axis=1)  → (B, H=32, T, Dh)
```

第 8 步在 RoPE *之后*重复是正确的，因为重复的头共享同一个绝对位置
—— 它们都属于同一个 token。在 RoPE 之前重复会错误地对不同副本应用不同的旋转。

### 因果遮罩：Prefill vs Decode

```
Prefill (S > 1):
  query_pos = [0, 1, 2, 3, 4, 5]  (offset=0)
  key_pos   = [0, 1, 2, 3, 4, 5]  (缓存中目前全部 key)
  mask      = key_pos > query_pos  → 下三角遮罩
  Token 3 只能关注位置 [0,1,2,3]。

Decode (S = 1):
  query_pos = [6]                  (offset=6)
  key_pos   = [0, 1, 2, 3, 4, 5]  (全部来自缓存)
  mask      = key_pos > query_pos  → 全部为 False，无需遮罩
  单个新 token 可以关注历史中的一切。
```

在 decode 阶段，`S=1` 意味着只有一个查询位置，且它排在所有缓存 key 之后
—— 它可以关注所有内容，根本不需要遮罩。代码仍然计算遮罩是为了正确性
（在 prefill 时不计算就会出错），但在 decode 时遮罩实际上是空操作。

---

## 总结：一次完整的生成请求

```
用户: "The capital of France is"

PREFILL（1 次前向传播）
  tokenize("The capital of France is") → [128000, 791, 14013, 315, 11405, 374]
  allocate KVCache(max_seq_len=2048)
  model(input_ids=(1,6), cache, position_offset=0)
    → 16 层在位置 0..5 写入 K/V
  cache.advance(6)
  mx.eval(final_logits) + cache.eval()
  sample → token "Paris" (3663)

DECODE LOOP（每个 token 一次前向传播）
  ┌─ step 0: token "Paris"
  │   model(input_ids=(1,1) [[3663]], cache, position_offset=6)
  │   mx.eval(logits) + cache.eval()
  │   cache.advance(1)
  │   sample → token ","
  │
  ├─ step 1: token ","
  │   model(input_ids=(1,1) [[11]], cache, position_offset=7)
  │   mx.eval(logits) + cache.eval()
  │   cache.advance(1)
  │   sample → token " located"
  │
  ├─ step 2: ...
  │   ...直到 EOS 或 max_new_tokens
  │
  └─ 逐个 yield 解码后的文本片段
```

循环在以下条件满足时停止：
- 采样到的 token 等于 `eos_token_id`（引擎*不会* yield EOS）
- 达到 `max_new_tokens`（引擎跳过最后浪费的一次 decode forward）
