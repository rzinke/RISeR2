"""
These modules deal with dated markers consisting of PDFs describing the
displacement and age of a geologic feature.
"""

# Public API
__all__ = [
    "DatedMarker",
    "readers",
]

# Import modules
from .DatedMarker import DatedMarker
from . import readers