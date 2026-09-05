# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
These functions enforce consistency in PDF value array (x-axis) formatting.
"""

# Public API
__all__ = [
    "sample_spacing_from_pdf",
    "sample_spacing_array_from_pdf",
    "value_array_params_from_pdfs",
    "precise_array",
    "check_pdfs_sampling",
]


# Import modules
import warnings

import numpy as np

from .. import precision
from .probability_density_function import ProbabilityDensityFunction as PDF


#################### SAMPLING LIMITS AND RATE ####################
def sample_spacing_from_pdf(pdf: PDF, verbose: bool = False) -> float:
    """Determine the average change in x (dx) for a discrete PDF.

    Parameters
    ----------
    pdf : PDF
        PDF from which to determine sample spacing.

    Returns
    -------
    dx : float
        Sample spacing (single value).
    """
    # Deteremine threshold for regular sampling based on CoV
    diff_x = np.diff(pdf.x)
    diff_x_std = np.std(diff_x)
    diff_x_cv = diff_x_std / np.abs(np.mean(diff_x))

    # Raise warning if a single value is not representative
    if diff_x_cv > 10 ** -precision.RISER_PRECISION:
        warnings.warn(
            f"Sample spacing varies by {diff_x_std}. "
            f"A single value might not be representative."
        )

    # Representative spacing value
    dx = precision.fix_precision(np.median(diff_x))

    # Report if requested
    if verbose:
        print(f"Sample spacing {dx}")

    return dx


def sample_spacing_array_from_pdf(
    pdf: PDF, verbose: bool = False
) -> np.ndarray:
    """Return an array of the changes in x (dx) for a discrete PDF.

    In classical calculus, dx is a single scalar number, which assumes that the
    function is regularly sampled.

    For the discrete PDFs used in practical applications, that might not be
    the case, i.e., the bin sizes may vary and a vector of bin sizes (dx's) is
    necessary.

    This routine returns a vector of bin sizes for all n-values in a PDF.
    The first n - 1 bin sizes are based on the differences from one x-value to
    the next. If the PDF is regularly sampled, the final bin size will be the
    same as the average bin size. If the PDF is irregularly sampled, the final
    bin size will be zero (excluding the final measurement).

    Parameters
    ----------
    pdf : PDF
        PDF for which to determine dx.

    Returns
    -------
    dx : np.ndarray
        dx values.
    """
    # Deteremine differences between x-samples
    diff_x = np.diff(pdf.x)

    # Determine regularity of sampling
    diff_x_std = np.std(diff_x)

    # Report if requested
    if verbose:
        print(f"Sample spacing mean {np.mean(diff_x)}, std {diff_x_std}")

    # Deteremine threshold for regular sampling based on CoV
    diff_x_cv = diff_x_std / np.abs(np.mean(diff_x))

    # Check regularity against machine error
    if diff_x_cv > 10 ** -precision.RISER_PRECISION:
        # Irregular sampling of PDF
        return precision.fix_precision(np.diff(pdf.x, append=pdf.x[-1]))
    else:
        # Regular sampling
        return precision.fix_precision(
            np.diff(pdf.x, append=pdf.x[-1] + np.mean(diff_x))
        )


def value_array_params_from_pdfs(
    pdfs: list[PDF], verbose: bool = False
) -> tuple[float, float, float]:
    """Determine the value limits for a set of PDFs.

    Parameters
    ----------
    pdfs : list[PDF]
        PDFs to resample.

    Returns
    -------
    xmin : float
        Minimum array value.
    xmax : float
        Maximum array value.
    dx : float
        Value array step.
    """
    # Initial values
    x0 = pdfs[0]
    xmin = x0.x.min()
    xmax = x0.x.max()
    dx = sample_spacing_from_pdf(pdfs[0])

    # Loop through subsequent PDFs
    for pdf in pdfs[1:]:
        # x-values for each PDF
        xi_min = pdf.x.min()
        xi_max = pdf.x.max()

        # Update min/max values
        xmin = xi_min if xi_min < xmin else xmin
        xmax = xi_max if xi_max > xmax else xmax

        # Sample spacing for each PDF
        dxi = sample_spacing_from_pdf(pdf)

        # Update dx value
        dx = dxi if dxi < dx else dx

    # Report if requested
    if verbose:
        print(f"xmin {xmin}, xmax {xmax}, dx {dx}")

    return xmin, xmax, dx


#################### VALUE ARRAYS ####################
def precise_array(
    xmin: float, xmax: float, dx: float, verbose: bool = False
) -> np.ndarray:
    """Create an array (vector) of values over the PDF domain.

    Parameters
    ----------
    xmin : float
        Minimum array value.
    xmax : float
        Maximum array value.
    dx : float
        Value array step.

    Returns
    -------
    x : np.ndarray
        Value array.
    """
    # Fix precision
    dx = precision.fix_precision(dx)

    # Determine number of samples
    n = np.round((xmax - xmin) / dx).astype(int) + 1

    # Create value array
    x = np.linspace(xmin, xmax, n)
    x = precision.fix_precision(x)

    # Report if requested
    if verbose:
        print(f"{len(x)} discrete values with {dx} spacing")

    return x


#################### CHECKS ####################
def check_pdfs_sampling(pdfs: list[PDF]) -> None:
    """Check that all PDFs are sampled over the same value array.

    Parameters
    ----------
    pdfs : list[PDF]
        PDFs to check.

    Returns
    -------
    None
    """
    # Initial value array
    x0 = pdfs[0].x

    # Loop through subsequent PDFs
    for pdf in pdfs[1:]:
        if not np.array_equal(pdf.x, x0):
            raise ValueError("Not all PDFs are sampled over same values")


# end of file
