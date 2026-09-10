# src/riser/variable_functions/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
Functions to carry out the transformation of random variables.
Transformation involves integrating over the values of one input, and weighting
by the density of the other input at the corresponding value that would produce
the output value.
"""

# Import modules
from . import arithmetic


# Public API
__all__ = [
    "arithmetic",
]


# end of file
