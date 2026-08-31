# src/riser/variable_pairs/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
These modules deal with dated markers consisting of two PDFs describing a pair
of observations, e.g., the displacement and age of a geologic feature.
"""

# Import modules
from .variable_pair import VariablePair
from .dated_marker import DatedMarker
from . import readers

# Public API
__all__ = (
    "VariablePair",
    "DatedMarker",
    "readers",
)


# end of file
