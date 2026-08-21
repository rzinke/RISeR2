# src/riser/sampling/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
These features deal with samples or realizations of a PDF.
pdf_formation handles conversion of samples into an empirical PDF.
filtering functions smooth the probability densities.
"""

# Public API
__all__ = [
    "filtering",
    "mc_sampling",
    "pdf_formation",
    "sample_statistics",
]

# Import modules
from . import filtering
from . import mc_sampling
from . import pdf_formation
from . import sample_statistics


# end of file
