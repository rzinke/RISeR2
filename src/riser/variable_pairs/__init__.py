# src/riser/variable_pairs/__init__.py
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
