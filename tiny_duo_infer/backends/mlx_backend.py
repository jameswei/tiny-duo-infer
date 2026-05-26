"""
MLX backend helper functions.

Provides thin wrappers around MLX for the Tier-1 operations defined in
backends/protocol.py. Phase 1 uses MLX directly in model code; these helpers
exist here so they can be cleanly extracted into a conforming Backend
implementation in Phase 2.

Key MLX behaviour to understand:
  MLX uses lazy evaluation — operations build a computation graph but do not
  execute until mx.eval() is called. Phase 1 synchronizes only at engine
  boundaries: after prefill, and once per decode step after the full forward
  pass. Do not call mx.eval() inside individual layers; that would add
  unnecessary GPU/CPU synchronization points and hide where inference steps
  begin and end.
"""
