# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

"""
Convert sample values to a pseudo-continuous PDF.
"""

# Public API
__all__ = [
    "PDF_FORMATION_METHODS",
    "get_pdf_formation_function",
]


# Import modules
from collections.abc import Callable

import numpy as np
import scipy as sp

from .. import probability_functions as PDFs


#################### FORMATION METHODS ####################
def samples_to_pdf_histogram(
    samples: np.ndarray,
    *,
    # Domain
    xmin: float | None = None,
    xmax: float | None = None,
    dx: float | None = None,
    # Metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> PDFs.PDF:
    """Form discrete samples into a PDF by binning them into a histogram.

    The histogram has one fewer values than bin edges, so the probability
    density is assigned to the bin edges as follows. Each interior bin edge
    takes the mean of the densities of the two bins it separates, and the first
    and last edges take the density of their adjacent bin. This conserves the
    probability mass of the histogram, and, unlike assigning each density to
    one edge of its bin, does not shift the PDF by half a bin. The first edge
    is placed at `xmin` so that the smallest values are preserved.

    Parameters
    ----------
    samples : np.ndarray
        Discrete samples from which to form the PDF.
    xmin : float, optional
        Minimum value to consider.
    xmax : float, optional
        Maximum value to consider.
    dx : float, optional
        Value array step.
        If `None`, the number of bins is computed from the samples within
        `xmin` and `xmax` using `np.histogram_bin_edges` with `bins="auto"`.
    name : str, optional
        Brief descriptive identifier for output PDF.
    variable_type : str, optional
        Sampled quantity for output PDF.
    unit : str, optional
        Value unit for output PDF.

    Returns
    -------
    pdf : PDF
        Empirical PDF based on samples.
    """
    if verbose:
        print("Converting samples to PDF using histogram method")

    # Determine value limits
    xmin = np.min(samples) if xmin is None else xmin
    xmax = np.max(samples) if xmax is None else xmax

    # Determine PDF domain array spacing from histogram bin sizes
    if dx is None:
        # Determine histogram bin edges based on samples and x-range
        bin_edges = np.histogram_bin_edges(
            samples, range=(xmin, xmax), bins="auto"
        )

        # Number of bins based on bin edges
        n_bins = len(bin_edges) - 1

        # PDF spacing
        dx = (xmax - xmin) / n_bins

    # Create histogram value array
    x = PDFs.value_arrays.precise_array(xmin, xmax, dx)

    # Bin points in histogram
    px, _ = np.histogram(samples, bins=x, density=True)

    # Assign bin densities to bin edges
    interior_densities = (px[:-1] + px[1:]) / 2
    px = np.concatenate([[px[0]], interior_densities, [px[-1]]])

    # Form histogram data into PDF
    pdf = PDFs.PDF(
        x=x,
        px=px,
        name=name,
        variable_type=variable_type,
        unit=unit,
    )

    return pdf


def samples_to_pdf_kde(
    samples: np.ndarray,
    *,
    # Domain
    xmin: float | None = None,
    xmax: float | None = None,
    dx: float | None = None,
    # Metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> PDFs.PDF:
    """Form discrete samples into a PDF using kernel density estimation (KDE)
    with a Gaussian kernel.

    This method is not recommended for converting slip rate samples into PDFs
    because slip rates skew toward faster values, and an adaptive kernel
    is necessary to adjust the bandwidth between more frequent samples at
    smaller values, to sparser samples at larger values.

    Parameters
    ----------
    samples : np.ndarray
        Discrete samples from which to form the PDF.
    xmin : float, optional
        Minimum value to consider.
    xmax : float, optional
        Maximum value to consider.
    dx : float, optional
        Value array step.
    name : str, optional
        Brief descriptive identifier for output PDF.
    variable_type : str, optional
        Sampled quantity for output PDF.
    unit : str, optional
        Value unit for output PDF.

    Returns
    -------
    pdf : PDF
        Empirical PDF based on samples.
    """
    if verbose:
        print("Converting samples to PDF using KDE")

    # Determine value limits
    xmin = np.min(samples) if xmin is None else xmin
    xmax = np.max(samples) if xmax is None else xmax

    # Determine bin sizes
    n_samples = len(samples)
    dx = 1 / np.sqrt(n_samples) if dx is None else dx

    # Create histogram value array
    x = PDFs.value_arrays.precise_array(xmin, xmax, dx)

    # Compute KDE
    kde = sp.stats.gaussian_kde(samples)

    # Interpolate along value array
    px = kde.pdf(x)

    # Form histogram data into PDF
    pdf = PDFs.PDF(
        x=x,
        px=px,
        name=name,
        variable_type=variable_type,
        unit=unit,
    )

    return pdf


PDF_FORMATION_METHODS = {
    "histogram": samples_to_pdf_histogram,
    "kde": samples_to_pdf_kde,
}


def get_pdf_formation_function(method: str, verbose: bool = False) -> Callable:
    """Retrieve a PDF formation function by name.

    Parameters
    ----------
    method : str
        PDF formation method.

    Returns
    -------
    Callable
        Function by which to form the PDF.
    """
    # Format method input
    method = method.lower()

    # Check that method is valid
    if method not in PDF_FORMATION_METHODS:
        raise ValueError(
            f"PDF formation method '{method}' not supported. "
            f"Use one of {', '.join(PDF_FORMATION_METHODS)}"
        )

    # Report if requested
    if verbose:
        print(f"PDF formation method: {method}")

    return PDF_FORMATION_METHODS[method]


# end of file
