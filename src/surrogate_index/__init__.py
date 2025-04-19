# SPDX-FileCopyrightText: 2025-present Kideok Kwon <kideokk16@gmail.com>
#
# SPDX-License-Identifier: MIT
from importlib.metadata import version

from .eif import SurrogateIndexEIF

__all__ = ["SurrogateIndexEIF"]
__version__ = version("surrogate-index")
