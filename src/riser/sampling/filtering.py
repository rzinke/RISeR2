# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.


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

# Public API
__all__ = [
    "FILTER_TYPES",
    "get_filter_by_name",
    "filter_pdf",
]


# Import modules
import copy
from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy as sp

from .. import probability_functions as PDFs


#################### FILTERS ####################
class FIRFilter:
    """Base class for a 1D FIR filter
    """

    filter_type: str | None = None

    def __init__(self, h: np.ndarray) -> None:
        """Initialize a generic FIRFilter.

        Parameters
        ----------
        h : np.ndarray
            Filter kernel.
        """
        # Filter values
        self.h = h.copy()

        # Ensure neutral gain
        self._normalize_gain_()

    def _normalize_gain_(self) -> None:
        """Ensure that the filter does not change the overall gain of the
        data series to which it applies.
        """
        # Scale sum to 1.0
        self.h /= np.sum(self.h)

    def __len__(self) -> int:
        return len(self.h)

    def __str__(self) -> str:
        print_str = f"{len(self.h)}-width {self.filter_type} filter"

        return print_str


class MeanFilter(FIRFilter):
    """Mean filter.
    """
    filter_type = "mean"

    def __init__(self, width: int) -> None:
        """Initialize a moving mean filter.

        Parameters
        ----------
        width : int
            Filter width in samples (dx units).
        """
        # Create basic filter values
        h = np.ones(width)

        # Format as FIRFilter object
        super().__init__(h)


class GaussFilter(FIRFilter):
    """Gauss filter.
    For small values, this will look similar to a triangle.
    """
    filter_type = "gaussian"

    def __init__(self, width: int) -> None:
        """Width is the total width.

        For a 2-sigma range, 1 sigma should be one half of half the width.

        Parameters
        ----------
        width : int
            Filter width in samples (dx units).
        """
        # Create basic filter values
        h = sp.signal.windows.gaussian(width, width / 4)

        # Format as FIRFilter object
        super().__init__(h)


FILTER_TYPES = {
    "mean": MeanFilter,
    "gaussian": GaussFilter,
}


def get_filter_by_name(
    filter_type: str, verbose: bool = False
) -> Callable[[int], FIRFilter]:
    """Retrieve an FIRFilter class by name.

    Parameters
    ----------
    filter_type : str
        Filter type.

    Returns
    -------
    FIRFilter
        Uninitialized filter of the specified type.
    """
    # Check filter specification is valid
    if filter_type not in FILTER_TYPES:
        raise ValueError(
            f"Filter type '{filter_type}' not valid. "
            f"Use one of {', '.join(FILTER_TYPES)}"
        )

    # Report if requested
    if verbose:
        print(f"Retrieving {filter_type} filter")

    return FILTER_TYPES[filter_type]


#################### FILTER APPLICATION ####################
def filter_pdf(
    pdf: PDFs.PDF,
    filter_type: str,
    filter_width: int,
    edge_padding: str = "zeros",
    preserve_edges: bool = False,
    verbose: bool = False,
) -> PDFs.PDF:
    """Apply a finite impulse response filter to the probability density
    values of a PDF.

    Total probability must be reset to 1.0 after filtering.

    Be wary of edge effects! Especially when filtering slip rates.
    Currently, edge effect mitigation involves padding the edge values with
    either zeros or edge values.

    Parameters
    ----------
    pdf : PDF
        PDF to filter.
    filter_type : str
        Filter type.
    filter_width : int
        Filter width in samples (dx units).
    edge_padding : str
        Method for padding to mitigate edge effects.
    """
    # Construct filter
    filt = get_filter_by_name(filter_type)(filter_width)

    # Report if requested
    if verbose:
        print(f"Applying {filt}")

    # Convert edge padding argument to padding mode
    padding_mode: Literal["constant", "edge"]
    if edge_padding == "zeros":
        padding_mode = "constant"
    elif edge_padding == "edges":
        padding_mode = "edge"
    else:
        raise ValueError(
            f"Edge padding '{edge_padding}' not supported. "
            f"Use one of 'zeros', 'edges'."
        )

    # Filter half-width
    w2 = filter_width // 2

    # Pad PDF
    px = np.pad(pdf.px, (w2, w2), padding_mode)

    # Apply filter to PDF
    px = sp.signal.convolve(px, filt.h, "same")

    # Trim padding
    px = px[w2:-w2]

    # Preserve edges if requested
    if preserve_edges:
        # Loop through edge values (output-side convolution)
        for i in range(w2):
            # Edge filter width
            w_edge = 2*i + 1

            # Re-formulate filter
            edge_filt = get_filter_by_name(filter_type)(w_edge)

            # Apply filter to front edge
            px[i] = np.sum(pdf.px[:w_edge] * edge_filt.h)

            # Apply filter to back edge
            px[-(i+1)] = np.sum(pdf.px[-w_edge:] * edge_filt.h)

    # Form results into PDF
    filt_pdf = PDFs.PDF(
        x=pdf.x,
        px=px,
        normalize_area=True,
        name=pdf.name,
        variable_type=pdf.variable_type,
        unit=pdf.unit,
    )

    return filt_pdf


# end of file
