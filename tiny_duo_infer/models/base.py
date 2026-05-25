"""
Base module infrastructure: Module ABC, Linear, Embedding.

Provides a minimal inference-only module system. We do not subclass
mlx.nn.Module or torch.nn.Module — this keeps model code backend-neutral
and makes the module system itself visible and understandable.

Weights are stored as plain backend-array attributes (mx.array in Phase 1).
No gradient tracking, no parameter registration, no device movement API:
inference only.

The load_weights() routing protocol:
  Each module receives a flat dict with dot-separated paths relative to itself.
  Direct keys (no dot) are set as attributes on self via setattr().
  Prefixed keys are split on the first dot and the remainder is forwarded
  recursively to the named sub-module attribute.

  Example for LlamaBlock.load_weights({
      "input_norm.weight": ...,       # → setattr(self.input_norm, "weight", ...)
      "attn.q_proj.weight": ...,      # → self.attn.load_weights({"q_proj.weight": ...})
      "attn.q_proj.weight": ...,      # → self.attn.q_proj.load_weights({"weight": ...})
  })
"""

from __future__ import annotations

import mlx.core as mx


class Module:
    """
    Base class for all model modules in this engine.

    Mirrors the minimal interface of mlx.nn.Module and torch.nn.Module without
    depending on either. Subclasses implement forward() and can override
    load_weights() for custom weight-loading logic; the default implementation
    handles the dot-path routing protocol described in this module's docstring.
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Delegate to forward()."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Subclasses implement the layer's computation here."""
        raise NotImplementedError(f"{type(self).__name__}.forward() is not implemented")

    def load_weights(self, weights: dict[str, mx.array]) -> None:
        """
        Populate weight attributes from a flat dict of backend arrays.

        Keys are dot-separated paths relative to this module.
        A key with no dot is set directly as an attribute on self.
        A key with a dot is split on the first dot: the left part names a
        sub-module attribute, and the right part is forwarded recursively.

        Args:
            weights: flat dict mapping dot-path key → mx.array.

        Raises:
            KeyError: if a dotted key names an attribute that is not a Module.
        """
        for key, value in weights.items():
            if "." not in key:
                setattr(self, key, value)
            else:
                attr_name, remainder = key.split(".", 1)
                sub_module = getattr(self, attr_name, None)
                if not isinstance(sub_module, Module):
                    raise KeyError(
                        f"{type(self).__name__}.load_weights: no Module sub-attribute "
                        f"{attr_name!r} found while routing key {key!r}"
                    )
                sub_module.load_weights({remainder: value})


class Linear(Module):
    """
    Linear projection: y = x @ weight.T

    No bias in Llama. Weight stored as (out_features, in_features) following
    HuggingFace convention; the .T transpose is applied on every forward call.

    This is a thin wrapper — the real learning is in the layers that use it
    (attention projections, FFN projections, lm_head).

    Attributes:
        weight: (out_features, in_features) mx.array, set by load_weights().
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Record dimensions and initialise weight to None.

        Weight is not allocated here — it is populated via load_weights()
        from the safetensors checkpoint.

        Args:
            in_features:  input dimension (columns of weight matrix).
            out_features: output dimension (rows of weight matrix).
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight: mx.array | None = None  # (out_features, in_features)

    def forward(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)  — computed as x @ weight.T
        """
        return x @ self.weight.T


class Embedding(Module):
    """
    Token embedding lookup table.

    Maps integer token IDs to dense embedding vectors by indexing into the
    weight matrix. This is the first operation in every forward pass.

    In Llama-3.2-1B the embedding weight is tied to lm_head: both point to
    the same underlying array. Tied embedding handling is done in
    llama_converter.py during weight loading — nothing special is needed here.

    Attributes:
        weight: (vocab_size, d_model) mx.array, set by load_weights().
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        Record dimensions and initialise weight to None.

        Args:
            vocab_size: number of vocabulary tokens (rows of weight matrix).
            d_model:    embedding dimension (columns of weight matrix).
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight: mx.array | None = None  # (vocab_size, d_model)

    def forward(self, token_ids: mx.array) -> mx.array:
        """
        Look up embedding vectors for each token ID.

        Args:
            token_ids: (B, S) integer token IDs.
        Returns:
            (B, S, d_model) embedding vectors.
        """
        return self.weight[token_ids]
