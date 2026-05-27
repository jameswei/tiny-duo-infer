"""
Tests for tiny_duo_infer.models.base: Module, Linear, Embedding.
Tests for LlamaBlock and LlamaModel assembly (stubs until P1-T11).

All tests use small synthetic tensors or TINY_CONFIG dimensions.
No real model artifacts required.

Test categories (Module):
  - __call__ delegates to forward()
  - forward() raises NotImplementedError on the base class
  - load_weights() sets direct attributes (no dot in key)
  - load_weights() routes dotted keys to sub-modules recursively
  - load_weights() raises KeyError for a dotted key whose prefix is not a Module

Test categories (Linear):
  - forward() shape: (..., in_features) → (..., out_features)
  - forward() values: y = x @ weight.T (no bias)
  - works with batch+sequence dimensions (3D input)
  - weight is set via load_weights()

Test categories (Embedding):
  - forward() shape: (B, S) → (B, S, d_model)
  - forward() values: returns the correct weight rows
  - weight is set via load_weights()
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from tiny_duo_infer.cache import KVCache
from tiny_duo_infer.layers.attention import Qwen3Attention
from tiny_duo_infer.models.base import Embedding, Linear, Module
from tiny_duo_infer.models.llama import LlamaBlock, LlamaModel
from tiny_duo_infer.models.qwen3 import Qwen3Block, Qwen3Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_array(*shape: int) -> mx.array:
    """Return a deterministic float32 array of the given shape."""
    return mx.random.normal(shape)


# ---------------------------------------------------------------------------
# Module — base class behaviour
# ---------------------------------------------------------------------------

class _IdentityModule(Module):
    """Minimal concrete Module that returns its input unchanged."""
    def forward(self, x: mx.array) -> mx.array:
        return x


class _ParentModule(Module):
    """Module with a named sub-module, for routing tests."""
    def __init__(self) -> None:
        self.child = _IdentityModule()

    def forward(self, x: mx.array) -> mx.array:
        return self.child(x)


def test_module_call_delegates_to_forward():
    m = _IdentityModule()
    x = make_array(3)
    result = m(x)
    assert result is x


def test_module_base_forward_raises():
    m = Module()
    with pytest.raises(NotImplementedError):
        m.forward()


def test_load_weights_sets_direct_attribute():
    """A key with no dot sets an attribute directly on self."""
    m = _IdentityModule()
    w = make_array(4)
    m.load_weights({"weight": w})
    assert m.weight is w


def test_load_weights_sets_multiple_direct_attributes():
    m = _IdentityModule()
    w1, w2 = make_array(2), make_array(3)
    m.load_weights({"weight": w1, "bias": w2})
    assert m.weight is w1
    assert m.bias is w2


def test_load_weights_routes_to_sub_module():
    """A dotted key routes the remainder to the named sub-module attribute."""
    p = _ParentModule()
    w = make_array(4)
    p.load_weights({"child.weight": w})
    assert p.child.weight is w


def test_load_weights_routes_recursively():
    """Multi-level dotted paths recurse through the module tree."""

    class GrandchildModule(Module):
        def forward(self, x: mx.array) -> mx.array:
            return x

    class ChildModule(Module):
        def __init__(self) -> None:
            self.grandchild = GrandchildModule()

        def forward(self, x: mx.array) -> mx.array:
            return self.grandchild(x)

    class RootModule(Module):
        def __init__(self) -> None:
            self.child = ChildModule()

        def forward(self, x: mx.array) -> mx.array:
            return self.child(x)

    root = RootModule()
    w = make_array(2)
    root.load_weights({"child.grandchild.weight": w})
    assert root.child.grandchild.weight is w


def test_load_weights_raises_when_sub_attr_is_not_module():
    """Routing to a non-Module attribute raises KeyError with a clear message."""
    m = _IdentityModule()
    m.not_a_module = 42  # plain int, not a Module
    with pytest.raises(KeyError, match="not_a_module"):
        m.load_weights({"not_a_module.weight": make_array(2)})


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def test_linear_forward_2d_shape():
    """(S, in) → (S, out)"""
    layer = Linear(4, 8)
    layer.weight = make_array(8, 4)
    out = layer(make_array(3, 4))
    assert out.shape == (3, 8)


def test_linear_forward_3d_shape():
    """(B, S, in) → (B, S, out) — the common attention projection case."""
    layer = Linear(16, 32)
    layer.weight = make_array(32, 16)
    out = layer(make_array(1, 5, 16))
    assert out.shape == (1, 5, 32)


def test_linear_forward_values():
    """y = x @ weight.T — verified with a hand-checkable example."""
    layer = Linear(2, 3)
    # weight: (3, 2) → weight.T: (2, 3)
    layer.weight = mx.array([[1.0, 0.0],
                              [0.0, 1.0],
                              [1.0, 1.0]])
    x = mx.array([[2.0, 3.0]])      # (1, 2)
    # expected: [[2*1+3*0, 2*0+3*1, 2*1+3*1]] = [[2, 3, 5]]
    expected = mx.array([[2.0, 3.0, 5.0]])
    mx.eval(layer(x), expected)
    assert mx.allclose(layer(x), expected).item()


def test_linear_weight_set_via_load_weights():
    """load_weights() assigns the weight attribute used by forward()."""
    layer = Linear(4, 8)
    w = make_array(8, 4)
    layer.load_weights({"weight": w})
    assert layer.weight is w


def test_linear_no_bias():
    """Linear has no bias: output is purely x @ weight.T."""
    layer = Linear(3, 3)
    # Identity weight: output should equal input
    layer.weight = mx.eye(3)
    x = mx.array([[1.0, 2.0, 3.0]])
    mx.eval(layer(x))
    assert mx.allclose(layer(x), x).item()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def test_embedding_forward_shape(tiny_config):
    """(B, S) token IDs → (B, S, d_model) embeddings."""
    vocab_size = tiny_config["vocab_size"]
    d_model = tiny_config["d_model"]
    emb = Embedding(vocab_size, d_model)
    emb.weight = make_array(vocab_size, d_model)
    token_ids = mx.array([[0, 1, 2, 3]])   # (1, 4)
    out = emb(token_ids)
    assert out.shape == (1, 4, d_model)


def test_embedding_forward_values():
    """Each token ID selects the corresponding row from the weight matrix."""
    emb = Embedding(vocab_size=4, d_model=3)
    emb.weight = mx.array([[1.0, 2.0, 3.0],    # id 0
                            [4.0, 5.0, 6.0],    # id 1
                            [7.0, 8.0, 9.0],    # id 2
                            [10., 11., 12.]])    # id 3
    token_ids = mx.array([[2, 0]])              # (1, 2)
    expected = mx.array([[[7.0, 8.0, 9.0],
                           [1.0, 2.0, 3.0]]])   # (1, 2, 3)
    mx.eval(emb(token_ids), expected)
    assert mx.allclose(emb(token_ids), expected).item()


def test_embedding_weight_set_via_load_weights():
    """load_weights() assigns the weight attribute used by forward()."""
    emb = Embedding(vocab_size=16, d_model=8)
    w = make_array(16, 8)
    emb.load_weights({"weight": w})
    assert emb.weight is w


def test_embedding_different_batch_sizes(tiny_config):
    """Embedding handles any (B, S) shape."""
    vocab_size = tiny_config["vocab_size"]
    d_model = tiny_config["d_model"]
    emb = Embedding(vocab_size, d_model)
    emb.weight = make_array(vocab_size, d_model)
    for b, s in [(1, 1), (1, 8), (2, 4)]:
        ids = mx.zeros((b, s), dtype=mx.int32)
        out = emb(ids)
        assert out.shape == (b, s, d_model)


# ---------------------------------------------------------------------------
# LlamaModel assembly
# ---------------------------------------------------------------------------

def _random_weights(config) -> dict:
    """Build a complete flat weight dict with random values matching config dims."""
    D, V, I = config.d_model, config.vocab_size, config.intermediate_size
    H, Hkv, Dh = config.n_heads, config.n_kv_heads, config.head_dim
    weights = {
        "embed_tokens.weight": mx.random.normal((V, D)),
        "final_norm.weight":   mx.random.normal((D,)),
        "lm_head.weight":      mx.random.normal((V, D)),
    }
    for i in range(config.n_layers):
        weights.update({
            f"layers.{i}.input_norm.weight":     mx.random.normal((D,)),
            f"layers.{i}.attn.q_proj.weight":    mx.random.normal((H * Dh, D)),
            f"layers.{i}.attn.k_proj.weight":    mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.v_proj.weight":    mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.o_proj.weight":    mx.random.normal((D, H * Dh)),
            f"layers.{i}.post_attn_norm.weight": mx.random.normal((D,)),
            f"layers.{i}.ffn.gate_proj.weight":  mx.random.normal((I, D)),
            f"layers.{i}.ffn.up_proj.weight":    mx.random.normal((I, D)),
            f"layers.{i}.ffn.down_proj.weight":  mx.random.normal((D, I)),
        })
    return weights


def _make_model(config) -> LlamaModel:
    model = LlamaModel(config)
    model.load_weights(_random_weights(config))
    return model


def _make_cache(config) -> KVCache:
    return KVCache(
        n_layers=config.n_layers,
        n_kv_heads=config.n_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=config.head_dim,
    )


def test_llama_model_prefill_logits_shape(tiny_model_config):
    """Prefill returns logits (B, S, V) for a prompt of length S."""
    S = 5
    model = _make_model(tiny_model_config)
    cache = _make_cache(tiny_model_config)
    input_ids = mx.array([[0, 1, 2, 3, 4]])  # (1, S)

    logits = model(input_ids, cache, position_offset=0)
    mx.eval(logits)

    assert logits.shape == (1, S, tiny_model_config.vocab_size)


def test_llama_model_decode_logits_shape(tiny_model_config):
    """Decode step (S=1) returns (B, 1, V) logits."""
    model = _make_model(tiny_model_config)
    cache = _make_cache(tiny_model_config)

    # Simulate prefill of 4 tokens, then one decode step
    prompt = mx.array([[0, 1, 2, 3]])  # (1, 4)
    model(prompt, cache, position_offset=0)
    cache.advance(4)

    token = mx.array([[5]])  # (1, 1)
    logits = model(token, cache, position_offset=cache.current_len)
    mx.eval(logits)

    assert logits.shape == (1, 1, tiny_model_config.vocab_size)


def test_llama_model_kv_cache_filled_during_prefill(tiny_model_config):
    """After forward over S tokens, all layers have KV data at positions [0, S)."""
    S = 3
    model = _make_model(tiny_model_config)
    cache = _make_cache(tiny_model_config)
    input_ids = mx.array([[0, 1, 2]])

    model(input_ids, cache, position_offset=0)
    mx.eval(cache._keys[0])

    # Model only writes via cache.update(); current_len stays 0 until engine calls advance()
    assert cache.current_len == 0

    # Manually advance (engine responsibility) and verify length
    cache.advance(S)
    assert cache.current_len == S


def test_llama_model_cache_current_len_unchanged_by_forward(tiny_model_config):
    """LlamaModel never calls cache.advance(); that is the engine's responsibility."""
    model = _make_model(tiny_model_config)
    cache = _make_cache(tiny_model_config)

    model(mx.array([[0, 1, 2, 3, 4]]), cache, position_offset=0)
    mx.eval(cache._keys[0])

    assert cache.current_len == 0


def test_llama_block_forward_shape(tiny_model_config):
    """LlamaBlock preserves (B, S, D) through both residual sub-layers."""
    from tiny_duo_infer.layers.rope import precompute_freqs
    cos_sin = precompute_freqs(
        tiny_model_config.head_dim,
        tiny_model_config.max_seq_len,
        tiny_model_config.rope_theta,
    )
    block = LlamaBlock(tiny_model_config, layer_idx=0, cos_sin=cos_sin)

    D = tiny_model_config.d_model
    Hkv, Dh = tiny_model_config.n_kv_heads, tiny_model_config.head_dim
    I = tiny_model_config.intermediate_size
    H = tiny_model_config.n_heads
    # Set random weights on all sub-modules
    block.input_norm.weight     = mx.ones((D,))
    block.attn.q_proj.weight    = mx.random.normal((H * Dh, D))
    block.attn.k_proj.weight    = mx.random.normal((Hkv * Dh, D))
    block.attn.v_proj.weight    = mx.random.normal((Hkv * Dh, D))
    block.attn.o_proj.weight    = mx.random.normal((D, H * Dh))
    block.post_attn_norm.weight = mx.ones((D,))
    block.ffn.gate_proj.weight  = mx.random.normal((I, D))
    block.ffn.up_proj.weight    = mx.random.normal((I, D))
    block.ffn.down_proj.weight  = mx.random.normal((D, I))

    cache = _make_cache(tiny_model_config)
    S = 4
    x = mx.random.normal((1, S, D))
    out = block(x, cache, layer_idx=0, position_offset=0)
    mx.eval(out)

    assert out.shape == x.shape


def test_llama_model_load_weights_routes_all_keys(tiny_model_config):
    """load_weights() correctly populates all sub-module weights including layers list."""
    model = LlamaModel(tiny_model_config)
    weights = _random_weights(tiny_model_config)
    model.load_weights(weights)

    assert model.embed_tokens.weight is weights["embed_tokens.weight"]
    assert model.final_norm.weight   is weights["final_norm.weight"]
    assert model.lm_head.weight      is weights["lm_head.weight"]

    for i in range(tiny_model_config.n_layers):
        assert model.layers[i].input_norm.weight     is weights[f"layers.{i}.input_norm.weight"]
        assert model.layers[i].attn.q_proj.weight    is weights[f"layers.{i}.attn.q_proj.weight"]
        assert model.layers[i].ffn.gate_proj.weight  is weights[f"layers.{i}.ffn.gate_proj.weight"]
        assert model.layers[i].ffn.down_proj.weight  is weights[f"layers.{i}.ffn.down_proj.weight"]


# ---------------------------------------------------------------------------
# Qwen3Model assembly
# ---------------------------------------------------------------------------

def _random_qwen3_weights(config) -> dict:
    """Build a complete flat Qwen3 weight dict with random values."""
    D, V, I = config.d_model, config.vocab_size, config.intermediate_size
    H, Hkv, Dh = config.n_heads, config.n_kv_heads, config.head_dim
    A = H * Dh
    weights = {
        "embed_tokens.weight": mx.random.normal((V, D)),
        "final_norm.weight": mx.random.normal((D,)),
        "lm_head.weight": mx.random.normal((V, D)),
    }
    for i in range(config.n_layers):
        weights.update({
            f"layers.{i}.input_norm.weight": mx.random.normal((D,)),
            f"layers.{i}.attn.q_proj.weight": mx.random.normal((A, D)),
            f"layers.{i}.attn.k_proj.weight": mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.v_proj.weight": mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.o_proj.weight": mx.random.normal((D, A)),
            f"layers.{i}.attn.q_norm.weight": mx.random.normal((Dh,)),
            f"layers.{i}.attn.k_norm.weight": mx.random.normal((Dh,)),
            f"layers.{i}.post_attn_norm.weight": mx.random.normal((D,)),
            f"layers.{i}.ffn.gate_proj.weight": mx.random.normal((I, D)),
            f"layers.{i}.ffn.up_proj.weight": mx.random.normal((I, D)),
            f"layers.{i}.ffn.down_proj.weight": mx.random.normal((D, I)),
        })
    return weights


def _make_qwen3_model(config) -> Qwen3Model:
    model = Qwen3Model(config)
    model.load_weights(_random_qwen3_weights(config))
    return model


def test_qwen3_block_uses_qwen3_attention(tiny_qwen3_model_config):
    """Qwen3Block instantiates Qwen3Attention, keeping model-family logic explicit."""
    from tiny_duo_infer.layers.rope import precompute_freqs

    cos_sin = precompute_freqs(
        tiny_qwen3_model_config.head_dim,
        tiny_qwen3_model_config.max_seq_len,
        tiny_qwen3_model_config.rope_theta,
    )
    block = Qwen3Block(tiny_qwen3_model_config, layer_idx=0, cos_sin=cos_sin)

    assert isinstance(block.attn, Qwen3Attention)
    assert block.attn.q_proj.out_features == (
        tiny_qwen3_model_config.n_heads * tiny_qwen3_model_config.head_dim
    )


def test_qwen3_model_prefill_logits_shape(tiny_qwen3_model_config):
    """Qwen3Model prefill returns logits (B, S, V) with A = H * Dh != D."""
    S = 5
    model = _make_qwen3_model(tiny_qwen3_model_config)
    cache = _make_cache(tiny_qwen3_model_config)
    input_ids = mx.array([[0, 1, 2, 3, 4]])  # (1, S)

    logits = model(input_ids, cache, position_offset=0)
    mx.eval(logits)

    assert tiny_qwen3_model_config.n_heads * tiny_qwen3_model_config.head_dim != (
        tiny_qwen3_model_config.d_model
    )
    assert logits.shape == (1, S, tiny_qwen3_model_config.vocab_size)


def test_qwen3_model_decode_logits_shape(tiny_qwen3_model_config):
    """Qwen3Model decode step returns (B, 1, V) after prefill."""
    model = _make_qwen3_model(tiny_qwen3_model_config)
    cache = _make_cache(tiny_qwen3_model_config)

    prompt = mx.array([[0, 1, 2]])
    model(prompt, cache, position_offset=0)
    cache.advance(3)

    token = mx.array([[4]])
    logits = model(token, cache, position_offset=cache.current_len)
    mx.eval(logits)

    assert logits.shape == (1, 1, tiny_qwen3_model_config.vocab_size)


def test_qwen3_model_cache_current_len_unchanged_by_forward(tiny_qwen3_model_config):
    """Qwen3Model leaves cache.advance() responsibility with the engine."""
    model = _make_qwen3_model(tiny_qwen3_model_config)
    cache = _make_cache(tiny_qwen3_model_config)

    model(mx.array([[0, 1, 2]]), cache, position_offset=0)
    mx.eval(cache._keys[0])

    assert cache.current_len == 0


def test_qwen3_model_load_weights_routes_all_keys(tiny_qwen3_model_config):
    """load_weights() populates Qwen3 layer weights, including q_norm/k_norm."""
    model = Qwen3Model(tiny_qwen3_model_config)
    weights = _random_qwen3_weights(tiny_qwen3_model_config)
    model.load_weights(weights)

    assert model.embed_tokens.weight is weights["embed_tokens.weight"]
    assert model.final_norm.weight is weights["final_norm.weight"]
    assert model.lm_head.weight is weights["lm_head.weight"]

    for i in range(tiny_qwen3_model_config.n_layers):
        assert (
            model.layers[i].input_norm.weight
            is weights[f"layers.{i}.input_norm.weight"]
        )
        assert (
            model.layers[i].attn.q_proj.weight
            is weights[f"layers.{i}.attn.q_proj.weight"]
        )
        assert (
            model.layers[i].attn.q_norm.weight
            is weights[f"layers.{i}.attn.q_norm.weight"]
        )
        assert (
            model.layers[i].attn.k_norm.weight
            is weights[f"layers.{i}.attn.k_norm.weight"]
        )
        assert (
            model.layers[i].ffn.down_proj.weight
            is weights[f"layers.{i}.ffn.down_proj.weight"]
        )


@pytest.mark.slow
def test_model_forward_smoke():
    """Load real Llama-3.2-1B weights and run a forward pass; verify logits shape."""
    pytest.skip("requires real model artifacts — run with --run-slow")
