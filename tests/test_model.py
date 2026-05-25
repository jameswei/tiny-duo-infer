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

from tiny_duo_infer.models.base import Embedding, Linear, Module


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
# LlamaModel assembly tests (stubs until P1-T11)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_model_forward_smoke():
    """Load real weights and run a forward pass; verify logits shape."""
    pytest.skip("not yet implemented")
