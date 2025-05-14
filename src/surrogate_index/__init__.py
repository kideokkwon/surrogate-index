# SPDX-FileCopyrightText: 2025-present Kideok Kwon <kideokk16@gmail.com>
# SPDX-License-Identifier: MIT
"""
Surrogate-Index EIF
~~~~~~~~~~~~~~~~~~~
Top-level package that exposes the efficient-influence-function helpers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------
try:
    __version__ = _pkg_version("surrogate-index")
except PackageNotFoundError:  # package is running from source tree
    __version__ = "0.0.0.dev0"

# ---------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------
from .eif import efficient_influence_function, output_inference  # noqa: F401

__all__: list[str] = [
    "efficient_influence_function",
    "output_inference",
    "__version__",
]
