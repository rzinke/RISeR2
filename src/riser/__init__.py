# src/riser/__init__.py
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
RISeR2: Rejection sampling for Incremental Slip Rate calculation.
"""

from importlib.metadata import version as _version


__version__ = _version("riser")


# Core modules (no internal dependency)
from . import (  # noqa: I001
    constants,
    integration,
    precision,
    units,
    variable_types,
)


# Subpackages (imported in dependency order)
from . import (
    probability_functions,
    variable_operations,
    variable_pairs,
    sampling,
    slip_rates,
    plotting,
)


# Commonly used classes
from .probability_functions import PDF
from .variable_pairs import DatedMarker, VariablePair


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
