# 深入解析：MLX 仅权重量化

本文聚焦 `tiny-duo-infer` 在 Phase 1.8 中实现的**仅权重量化（weight-only quantization）**。配套阅读：phase 规范 `docs/phases/phase-1.8-weight-quantization.md`，以及源码 `tiny_duo_infer/quantization.py`、`tiny_duo_infer/weights/quantizer.py`、`tiny_duo_infer/models/base.py`。

Phase 1.8 的目标不是写一个生产级的量化工具箱，而是把推理引擎中量化路径的每一处都暴露出来：哪些权重可以量化、打包整型权重如何取代浮点矩阵、融合 quantized matmul kernel 如何跳过 dequantize 步骤、以及如何**诚实地**核算出内存收益。

---

## 1. 为什么是仅权重量化

现代 transformer 权重很大。Llama-3.2-1B 在 bfloat16 下磁盘约 2.5 GB，加载之后线性投影权重在内存里占主要部分：

```
linear weight bytes ≈ Σ_layer 4 × d_model × d_model
                    + Σ_layer 3 × d_model × intermediate_size
                    + lm_head_out × d_model
```

对 Llama-3.2-1B 来说，仅 `Linear` 权重（bfloat16）就约 1.2 GB。剩余部分是 KV cache、激活、embedding、RoPE 表与各种开销。

可压缩这部分内存的三类方法：

| 方法 | 压缩对象 | 决定时机 | Phase 1.8？ |
|---|---|---|---|
| **仅权重量化** | `Linear` 模块的矩阵权重 | 离线 / 加载期 | ✅ INT4 / INT8 |
| 激活量化 | 各层之间的隐状态 | 运行时按 token | ❌ |
| KV-cache 量化 | 缓存的 K/V 张量 | 运行时按层 | ❌ |

Phase 1.8 选择最克制的一项：**仅权重**。激活、embedding、RMSNorm/Q-K-norm 权重、RoPE 表、KV cache 全都保持全精度。这让改动可以一层一层 review，并允许我们只**测量一个变量**：linear weight memory。

它也保留了激活精度——前向中的动态范围损失与全精度路径完全一致。**只有**每个 `Linear` 投影的矩阵乘是从压缩存储中取数。

---

## 2. 仿射量化，按组进行

Phase 1.8 的量化模式是 `"affine"`。沿输入维度，每 `group_size` 个相邻元素共享一个浮点 `scale` 与一个浮点 `bias`（zero-point）：

$$
q = \mathrm{round}\!\left(\frac{x - \mathrm{bias}}{\mathrm{scale}}\right),
\qquad
x \approx q \cdot \mathrm{scale} + \mathrm{bias}
$$

对形状为 `(out_features, in_features)` 的权重，`mx.quantize(weight, group_size, bits)` 会返回三个数组：

```text
qweight: (out_features, in_features * bits / 32)    打包整型
scales:  (out_features, in_features / group_size)   每组浮点 scale
biases:  (out_features, in_features / group_size)   每组浮点 bias
```

`qweight` 的打包规则是"沿输入维度，把尽可能多的 `bits` 宽整型塞进每个 32 位字"。`bits=4` 时 `in_features` 缩 8 倍；`bits=8` 时缩 4 倍。

每组的两个重建参数以 float 存储，数量随 `in_features / group_size` 增长。这就是 `group_size` 调控的权衡：组越小，重建越精细（质量更好），但 `scales`/`biases` 占的字节越多。

### 为什么 group_size 必须沿输入维度对齐

融合 kernel 一次算一个输出坐标，需要把 `in_features` 个贡献相加。每一个贡献来自某一组——而那一组要 dequantize，必须**恰好一对** `(scale, bias)`。如果 `in_features` 不被 `group_size` 整除，最后那个不完整组里的部分贡献就没有 scale/bias 可用，kernel 无法运行。

`tiny_duo_infer.quantization.QuantizationConfig` 在构造时校验这一点；`weights/quantizer.py` 在每个权重上再次复核，并抛出 `ValueError`，message 中点名违例的 key、`in_features`、`group_size`，从而避免半加载状态下的失败。

默认 `group_size = 64`。它能覆盖真实 Llama-3.2-1B 与 Qwen3-0.6B 的所有 `Linear` 矩阵——它们的相关输入维度（`d_model`、`intermediate_size`、attention 宽度）全是 64 的倍数。但 `d_model = 32` 的极小合成测试 fixture 用不了这个默认值，必须使用 `group_size = 32`（或 32 的其他因子）。Phase 1.8 spec 明确指出了这一点；集成测试同时验证成功用例与拒绝用例；CLI / profiling 入口都暴露 `--quant-group-size N` 让用户主动调节。

---

## 3. 融合的 quantized matmul

`Linear.forward()` 在运行时的分发路径：

```python
if isinstance(self.weight, QuantizedWeight):
    qw = self.weight
    return mx.quantized_matmul(
        x, qw.qweight, qw.scales, qw.biases,
        transpose=True,
        group_size=qw.group_size,
        bits=qw.bits,
        mode=qw.mode,
    )
return x @ self.weight.T
```

融合 kernel 概念上做的事：

1. 遍历输出坐标 `o ∈ [0, out_features)`；
2. 对沿输入维度的每个组，用 `scale[o, g]` 和 `bias[o, g]` 把 `group_size` 个整型 dequantize 进一个寄存器大小的 tile；
3. 累加 `dot(x_tile, dequantized_tile)` 到输出；
4. 输出 `y[..., o]`。

**对 Phase 1.8 来说至关重要**：**全精度权重矩阵从未在 DRAM 中物化。** 只有寄存器大小的 tile 被一次性 dequantize、用过即丢。这就是压缩存储的内存收益**在运行时也能保住**的根本原因——而不仅仅是加载期。

### 为什么 `mx.dequantize()` 受限

`mx.dequantize()` 能从 `(qweight, scales, biases)` 重建出全精度权重矩阵。它在测试和数值精度调试中很有用：

- 用一个量化权重计算"参考"全精度输出；
- 在数值容差内对比量化输出与参考输出；
- 检查哪些组在重建时贡献了最大的误差。

但如果运行时路径变成 `dequantize → matmul`，每次前向都会绕回全精度，内存收益就化为乌有。Phase 1.8 spec 明文禁止把它作为 normal path，架构 review gate 强制执行这一点。**加载期**的 eager dequantization 也基于同样原因被拒绝。

---

## 4. 资格判定：哪些权重会被量化

`weights/quantizer.py:_is_eligible()` 把全部资格规则用代码完整表达：

```python
def _is_eligible(key: str, tensor: mx.array) -> bool:
    if tensor.ndim != 2:
        return False
    if key in _ELIGIBLE_EXACT:        # {"lm_head.weight"}
        return True
    return any(key.endswith(suffix) for suffix in _ELIGIBLE_SUFFIXES)
```

`_ELIGIBLE_SUFFIXES` 是 7 个 `Linear` 投影后缀：

```
.q_proj.weight, .k_proj.weight, .v_proj.weight, .o_proj.weight,
.gate_proj.weight, .up_proj.weight, .down_proj.weight
```

两个过滤器覆盖了所有**不应**被量化的权重：

- `tensor.ndim != 2` 排除 RMSNorm 权重以及 Qwen3 的 `q_norm` / `k_norm` 权重——它们是 1-D `(head_dim,)`。
- 后缀列表是封闭的：不在表中的一律保持全精度。`embed_tokens.weight` 故意没有列入。

### Llama 的 tied embeddings

Llama-3.2-1B 把 `lm_head.weight` 和 `embed_tokens.weight` 绑在一起：转换器输出的两个 key 引用同一个 `mx.array` 对象。量化 `lm_head.weight` **绝不能**反向影响 `embed_tokens.weight`，因为：

- embedding 的访问方式是行索引——它不被矩阵乘消费，融合 quantized kernel 在运行时无法服务它。
- 把 embedding 当作量化的，需要在 `Embedding.forward()` 内引入另一条代码路径——Phase 1.8 不引入。

`weights/quantizer.py` 构造一个**新字典**，避免修改 `embed_tokens.weight` 下的数组对象。`quantize_weights()` 之后，`embed_tokens.weight` 仍是全精度 `mx.array`，`lm_head.weight` 是 `QuantizedWeight`。两个 key 不再共享对象——*逻辑上*的 tying 变为运行时约定，而不再是内存层别名。

### Qwen3 的 lm_head

尽管 Qwen3 的 config 标榜 tied embeddings，但其 safetensors 中**显式存储**了 `lm_head.weight`，因此转换器校验本来就期望该 key 存在。Phase 1.8 不改这一点——Qwen3 的 `lm_head.weight` 与其他权重一样直接成为量化投影。Qwen3 的 `q_norm.weight` 与 `k_norm.weight` 仍保持 1-D 全精度。

---

## 5. 内存核算

`weights/quantizer.py` 中的 `LinearWeightStats` 用四个数字总结 Phase 1.8 关心的对比：

```python
@dataclass
class LinearWeightStats:
    quantized_linear_count: int
    full_precision_linear_count: int
    linear_weight_full_precision_bytes: int
    linear_weight_runtime_bytes: int
```

`compute_linear_weight_stats()` 遍历 `quantize_weights()` 产出的字典，**只**统计可量化的 linear 投影。embedding、RMSNorm、Qwen3 Q/K-norm 全部排除——这样比较只回答一个范围明确的问题：

> **本 phase 允许压缩的那些 linear 权重，运行时实际占多少字节？全精度版本会占多少字节？**

这一选择写在 dataclass 的 docstring 里，并通过 `GenerationStats` 暴露出来。`GenerationStats` 中新增的 7 个量化字段：

| 字段 | 含义 |
|---|---|
| `quantization_mode` | `"none"`、`"int8"` 或 `"int4"` |
| `quantization_bits` | `None`、`8` 或 `4` |
| `quantization_group_size` | `None` 或所配置的 group size |
| `quantized_linear_count` | 多少个 `Linear` 模块持有 `QuantizedWeight` |
| `full_precision_linear_count` | 多少个 `Linear` 模块仍是全精度 |
| `linear_weight_full_precision_bytes` | 如果所有被计数的权重保持全精度时占的字节 |
| `linear_weight_runtime_bytes` | 这些权重在运行时**实际**占的字节 |

当量化被关闭时，这些字段仍然填充：`mode="none"`、`bits=None`、`group_size=None`、量化数为 0、`runtime_bytes == full_precision_bytes`。这样下游就能保证**"无量化"与"缺 stats 块"在响应里不可区分**——每条响应都携带完整 schema。

### 为什么吞吐不是硬性 gate

Phase 1.8 的验收门槛是 **linear weight memory 的下降**，不是 decode tokens/sec 的提升。原因：

- MLX 量化 kernel 的性能受 bit width、group size、矩阵形状、硬件状态影响很大。
- 小模型在全精度下可能是带宽受限；量化后变成算力受限——即使字节下降，wall-clock 时间未必更好。
- 一个学习优先的项目不应把"性能数字"当承诺指标。

因此 spec 要求：吞吐必须**测量并记录**，而不是**变好**。Profiling JSON v2 把数字呈现给读者，解读交给读者自己。

---

## 6. 加载流水线

Phase 1.8 spec 把模型加载形式化为四步：

1. 加载 safetensors；
2. 转换 HF key 并校验形状；
3. 量化合格的 project 权重（**新增的一步**）；
4. 把权重值灌入模型树。

`Engine.from_model_path()` 实现这四步，并多了一步**内存核算**——它**从扁平 project 权重字典**计算，发生在模型构造之前。这种顺序是有意的：`compute_linear_weight_stats()` 是字典遍历，不依赖任何已构造的模型；早做能让核算不依赖于 `Module.load_weights()`。

```text
1. 加载 safetensors             (weights/loader.py)
        │
        ▼
2. HF → project key 转换         (weights/llama_converter.py 或 qwen3_converter.py)
        │ 校验形状；通过对象同一性保留 tied embeddings
        ▼
3. 量化合格权重                  (weights/quantizer.py)   ← Phase 1.8 新步
        │ 对每个匹配资格规则的 Linear 投影矩阵调用 mx.quantize()，
        │ 其余原样保留；返回的是新字典——tied lm_head/embed_tokens
        │ 不再别名同一对象。
        ▼
3b. compute_linear_weight_stats(project_weights)
        │ 遍历扁平字典，仅统计合格的 Linear 权重。
        │ *在模型构造之前*运行，让核算从字典形态导出，不依赖
        │ Linear 模块的内省。结果存到 Engine 上，附加到每个
        │ GenerationStats 响应里。
        ▼
4. 构造模型 + 灌入权重           (Module.load_weights())
        │ model_cls(runtime_config) 先建出空模块树；之后按 key
        │ 设置 Linear.weight。Linear.weight 既可以是 mx.array
        │ 也可以是 QuantizedWeight；forward() 分发完全基于
        │ self.weight 的类型。
        ▼
5. return cls(model, tokenizer, ..., quantization, linear_weight_stats)
```

`Engine.from_model_path(model_path, max_seq_len, quantization=None)` 跑完整条流水。`quantization=None` 会让第 3 步短路——全精度加载是默认；第 3b 步仍照常对未改动的字典运行，所以"无量化"这条路径上 stats 字段同样被填充。传入一个 `QuantizationConfig` 则会让每个合格投影的 `Linear.forward()` 翻到量化分支，其它模块不变。

CLI、HTTP 服务启动、profiling 入口三处都接受 `--quantization {none,int4,int8}` 与 `--quant-group-size N`，构造 `QuantizationConfig` 并通过同一个 `from_model_path` 构造器转发。校验在早期就跑——非法 bit width、group size、不可整除的矩阵形状会在分配任何张量之前就失败。

---

## 7. 阅读代码时要写下的清单

针对每个组件，写下：

- **输入：** project 权重字典、`QuantizationConfig` 字段。
- **输出：** `dict[str, mx.array | QuantizedWeight]`、`LinearWeightStats`、`GenerationStats` 量化字段。
- **张量形状：** 源权重 `(out_features, in_features)`；`qweight: (out, in * bits / 32)`、`scales: (out, in / group_size)`、`biases: (out, in / group_size)`。
- **状态：** 哪个字典持有全精度数组、哪个持有 `QuantizedWeight`；哪些 `Linear` 实例接受哪种 weight 类型。
- **不变量：** `quantization_mode == "none"` ⇔ `bits is None and group_size is None`；量化关闭时，被计数的 linear 权重满足 `linear_weight_runtime_bytes == linear_weight_full_precision_bytes`。
- **失败情况：** `in_features` 不可整除、非法 `bits`、非法 `group_size`、不支持的 `mode`、eager dequantization（被 spec 禁止，而不是被代码禁止）。
- **静默生成 corruption 的危险点：** 把融合 kernel 偷换成 `dequantize → matmul`。生成仍然能跑，输出看起来仍然合理，**但内存收益悄无声息地消失**，`linear_weight_runtime_bytes` 不再反映 DRAM 中真实持有的字节数。

最后这一项正是集成测试 `test_llama_quantized_respects_stop_string` 选择**包裹真实量化模型**而非替换它的原因：它强制让 quantized matmul 路径执行，并断言正确的停止语义与非零的 `quantized_linear_count`——任何让其中一边失效的回归，都会被这一个测试同时捕获。

---

## 延伸阅读

- `docs/phases/phase-1.8-weight-quantization.md`：phase 权威 spec，含 out-of-scope 清单。
- `tiny_duo_infer/quantization.py`：`QuantizationConfig` 与 `QuantizedWeight`，含字段级 docstring。
- `tiny_duo_infer/weights/quantizer.py`：`quantize_weights()`、`LinearWeightStats`、`compute_linear_weight_stats()`。
- `tiny_duo_infer/models/base.py`：`Linear.forward()` 量化与全精度的分发。
- `tests/test_quantization.py`：config 校验、打包权重元数据、分发正确性。
- `tests/test_quantization_integration.py`：tiny Llama / Qwen3 在全精度、INT8、INT4 三条路径上的端到端生成。
- `learning_materials/roadmap.md`：包含 Phase 1.8 节的引导阅读顺序。
