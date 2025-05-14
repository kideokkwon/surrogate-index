# SPDX-FileCopyrightText: 2025-present Kideok Kwon <kideokk16@gmail.com>
# SPDX-License-Identifier: MIT
"""
Surrogate-Index EIF
~~~~~~~~~~~~~~~~~~~
Top-level package initializer for the surrogate-index module.

Exports:
- `efficient_influence_function`: Main estimator
- `output_inference`: Wrapper to summarize results
- `__version__`: Package version string
"""

from __future__ import annotations

import sys
from importlib.metadata import version as _get_version, PackageNotFoundError

# ---------------------------------------------------------------------
# Runtime Python version guard (requires 3.10+)
# ---------------------------------------------------------------------
if sys.version_info < (3, 10):
    raise RuntimeError("surrogate-index requires Python 3.10 or higher.")

# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------
try:
    __version__ = _get_version("surrogate-index")
except PackageNotFoundError:
    # When running from source (e.g., during development)
    __version__ = "0.0.0.dev0"

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
from .eif import efficient_influence_function, output_inference  # noqa: F401

__all__: list[str] = [
    "efficient_influence_function",
    "output_inference",
    "__version__",
]
