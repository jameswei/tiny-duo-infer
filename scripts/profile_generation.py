"""
Phase 1.7 profiling script entrypoint.

Thin wrapper around :func:`tiny_duo_infer.profiling.main`. The actual
implementation lives in the package module so it stays importable from
unit tests without polluting `sys.path` with `scripts/`.

Usage:
    uv run python scripts/profile_generation.py --help
"""

from __future__ import annotations

import sys

from tiny_duo_infer.profiling import main


if __name__ == "__main__":
    sys.exit(main())
