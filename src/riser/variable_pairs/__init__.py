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
from .readers import *

# Public API
__all__ = (
    variable_pair.__all__
    + dated_marker.__all__
    + readers.__all__
)


# end of file
