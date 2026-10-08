# src/riser/variable_pairs/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

"""
These modules deal with dated markers consisting of two PDFs describing a pair
of observations, e.g., the displacement and age of a geologic feature.
"""

# Import modules
from . import (
    interpolation,
    ordering,
    readers,
)
from .dated_marker import DatedMarker
from .variable_pair import VariablePair


# Public API
__all__ = (
    "VariablePair",
    "DatedMarker",
    "interpolation",
    "ordering",
    "readers",
)


# end of file
