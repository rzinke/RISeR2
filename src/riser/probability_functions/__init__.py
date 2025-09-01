"""
These features are built around probability density functions.
ProbabilityDensityFunction is the fundamental class upon with RISeR2 is built.
analytics quantify the properties of a single PDF.
readers contains functions for reading and saving PDFs from/to disk.
value_arrays contains functions for building and determining precise value
arrays.
interpolation handles resampling (interpolation/extrapolation) of PDF values
along a new value array.
parametric_functions includes definitions of parametric functions.
"""

from .ProbabilityDensityFunction import ProbabilityDensityFunction as PDF