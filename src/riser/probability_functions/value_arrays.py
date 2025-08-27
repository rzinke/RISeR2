# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Import modules
import warnings

import numpy as np

from riser import precision
from riser.probability_functions import PDF


#################### SAMPLING LIMITS AND RATE ####################
def sample_spacing_from_pdf(pdf:PDF, verbose=False) -> float:
    """Determine the average change in x (dx) for a discrete PDF.

    Args    pdf - PDF from which to determine sample spacing
    Returns dx - float, sample spacing (single value)
    """
    # Deteremine differences between x-samples
    diff_x = np.diff(pdf.x)

    # Determine regularity of sampling
    diff_x_std = np.std(diff_x)

    # Raise warning if a single value is not representative
    if diff_x_std > precision.RISER_PRECISION:
        warnings.warn(f"Sample spacing varies by {diff_x_std}. "
                      f"A single value might not be representative.")

    # Representative spacing value
    dx = precision.fix_precision(np.median(diff_x))

    # Report if requested
    if verbose == True:
        print(f"Sample spacing {dx}")

    return dx


def sample_spacing_array_from_pdf(pdf:PDF, verbose=False) -> np.ndarray:
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

    Args    pdf - PDF for which to determine dx
    Returns dx - np.ndarray
    """
    # Deteremine differences between x-samples
    diff_x = np.diff(pdf.x)

    # Determine regularity of sampling
    diff_x_std = np.std(diff_x)

    # Report if requested
    if verbose == True:
        print(f"Sample spacing mean {np.mean(diff_x)}, std {diff_x_std}")

    # Check regularity against machine error
    if diff_x_std > precision.RISER_PRECISION:
        # Irregular sampling of PDF
        return precision.fix_precision(np.diff(pdf.x, append=0))
    else:
        # Regular sampling
        return precision.fix_precision(np.diff(pdf.x, append=np.mean(diff_x)))


def value_array_params_from_pdfs(pdfs:list[PDF], verbose=False) -> \
        (float, float, float):
    """Determine the value limits for a set of PDFs.

    Args    pdfs - list[PDF], PDFs to resample
    Returns xmin, xmax - float, min/max values
            dx - float, value-step
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
    if verbose == True:
        print(f"xmin {xmin}, xmax {xmax}, dx {dx}")

    return xmin, xmax, dx


#################### VALUE ARRAYS ####################
def create_precise_value_array(xmin:float, xmax:float, dx:float,
        verbose=False) -> np.ndarray:
    """Create an array (vector) of values over the PDF domain.

    Args    xmin, xmax - float, min/max values
            dx - float, value-step
    Returns x - np.ndarray, value array
    """
    # Fix precision
    dx = precision.fix_precision(dx)

    # Create value array
    x = np.arange(xmin, xmax+dx, dx)
    x = precision.fix_precision(x)

    # Report if requested
    if verbose == True:
        print(f"{len(x)} discrete values with {dx} spacing")

    return x


#################### CHECKS ####################
def check_pdfs_sampling(pdfs:list[PDF]):
    """Check that all PDFs are sampled over the same value array.

    Args    pdfs - list[PDF], PDFs to check
    """
    # Initial value array
    x0 = pdfs[0].x

    # Loop through subsequent PDFs
    for pdf in pdfs[1:]:
        if not np.array_equal(pdf.x, x0):
            raise Exception("Not all PDFs are sampled over same values")


# end of file