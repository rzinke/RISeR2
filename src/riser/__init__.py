# src/riser/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
RISeR2: Rejection sampling for Incremental Slip Rate calculation.
"""

__version__ = "x.x.x"


# Core modules (no internal dependency)
from . import constants
from . import integration
from . import precision
from . import units
from . import variable_types


# Subpackages (imported in dependency order)
from . import probability_functions
from . import variable_operations
from . import variable_pairs
from . import sampling
from . import slip_rates
from . import plotting


# Commonly used classes
from .probability_functions import PDF
from .variable_pairs import VariablePair, DatedMarker


# Public API
__all__ = [
    # Core modules and subpackages
    "constants",
    "integration",
    "precision",
    "units",
    "variable_types",
    "probability_functions",
    "variable_operations",
    "sampling",
    "variable_pairs",
    "slip_rates",
    "plotting",
    # Commonly used classes
    "PDF",
    "VariablePair",
    "DatedMarker",
]


# end of file
