# SPDX-FileCopyrightText: 2025-present Kideok Kwon <kideokk16@gmail.com>
# SPDX-License-Identifier: MIT
"""
surrogate_index
~~~~~~~~~~~~~~~
Top-level package initialiser.

Public exports
--------------
- efficient_influence_function
- output_inference
- __version__
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------
# Python version guard
# ---------------------------------------------------------------------
if sys.version_info < (3, 10):
    raise RuntimeError("surrogate-index requires Python ≥3.10")

# ---------------------------------------------------------------------
# Version (single source of truth)
# ---------------------------------------------------------------------
from .__about__ import __version__  # noqa: E402

# ---------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------
from .eif import efficient_influence_function, output_inference  # noqa: F401,E402

__all__: list[str] = [
    "efficient_influence_function",
    "output_inference",
    "__version__",
]
