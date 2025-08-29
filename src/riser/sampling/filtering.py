# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


"""
Smooth rough or under-sampled PDFs.
Use of a finite impulse response (FIR) filter is appropriate, because two
common filter types, mean and gaussian, can be written as FIR filters.
Application by convolution is appropriate because it computes a sort of
weighted average based on the filter shape.

Implementation is currently carried out using classical convolution, which
assumes the input signal is stationary and infinite. This is not the case for
PDFs, and edge effects will be introduced where the filter interacts with the
other side of the PDF. This effect is relatively small for long-tailed PDFs,
but can be large for highly skewed PDFs such as those that commonly describe
slip rates.

Instead, an adaptive convolution that accounts for edges could be introduced
later.
"""


# Constants
FILTER_TYPES = [
    'mean',
    'gauss',
]


# Import modules
import copy

import numpy as np
import scipy as sp

from riser.probability_functions import PDF


#################### FILTERS ####################
class FIRFilter:
    """Base class for a 1D FIR filter
    """
    filter_type = None

    def __init__(self, h:np.ndarray):
        # Filter values
        self.h = h

        # Ensure neutral gain
        self.__check_gain__()

    def __check_gain__(self):
        """Ensure that the filter does not change the overall gain of the
        data series to which it applies.
        """
        # Scale sum to 1.0
        self.h /= np.sum(self.h)

        return True

    def __len__(self):
        return len(self.h)

    def __str__(self):
        print_str = f"{len(self.h)}-width {self.filter_type} filter"

        return print_str


class MeanFilter(FIRFilter):
    """Mean filter.
    """
    filter_type = "mean"

    def __init__(self, width:int):
        # Create basic filter values
        h = np.ones(width)

        # Format as FIRFilter object
        super().__init__(h)


class GaussFilter(FIRFilter):
    """Gauss filter.
    For small values, this will look similar to a triangle.
    """
    filter_type = "gauss"

    def __init__(self, width:int):
        """Width is the total width.
        For a 2-sigma range, 1 sigma should be one half of half the width.
        """
        # Create basic filter values
        h = sp.signal.windows.gaussian(width, width/4)

        # Format as FIRFilter object
        super().__init__(h)


def get_filter_by_name(filter_type:str, verbose=False):
    """Retrieve an FIRFilter class by name.
    """
    # Check filter specification is valid
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"Filter type not valid. "
                         f"Must be one of {FILTER_TYPES}")

    # Report if requested
    if verbose == True:
        print(f"Retrieving {filter_type} filter")

    # Return filter class
    if filter_type == "mean":
        return MeanFilter
    elif filter_type == "gauss":
        return GaussFilter


#################### FILTER APPLICATION ####################
def filter_pdf(pdf:PDF, filter_type:str, filter_width:int,
               edge_padding:str='zeros', verbose=False) -> PDF:
    """Apply a finite impulse response filter to the probability density
    values of a PDF.

    Total probability must be reset to 1.0 after filtering.

    Be wary of edge effects! Especially when filtering slip rates.
    Currently, edge effect mitigation involves padding the edge values with
    either zeros or edge values.

    Args    pdf - PDF to filter
            filter_type - str, filter type, e.g., mean
            filter_width - int, filter width in samples
    """
    # Construct filter
    filt = get_filter_by_name(filter_type)(filter_width)

    # Report if requested
    if verbose == True:
        print(f"Applying {filt}")

    # Filter half-width
    w2 = filter_width // 2

    # Pad PDF
    edge_padding = 'constant' if 'zero' in edge_padding else edge_padding
    px = np.pad(pdf.px, (w2, w2), edge_padding)

    # Apply filter to PDF
    px = sp.signal.convolve(px, filt.h, 'same')

    # Trim padding
    px = px[w2:-w2]

    # Form results into PDF
    pdf_args = {
        'x': pdf.x,
        'px': px,
        'normalize_area': True,
        'name': pdf.name,
        'variable_type': pdf.variable_type,
        'unit': pdf.unit,
    }
    filt_pdf = PDF(**pdf_args)

    return filt_pdf


# end of file