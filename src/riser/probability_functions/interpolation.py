# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


"""
Interpolation in this context is really resampling, because it includes
extrapolation, but the term is used to avoid confusion with Monte Carlo
sampling.

Values beyond the defined limits (domain) of the original PDF are assumed to
have zero probability density.

These routines are aimed toward resampling a PDF f(x) along a different set of
values that the random variable could take (e.g., a change in the sampling rate
or domain).
"""

# Import modules
import numpy as np

from riser.probability_functions import PDF
from riser.probability_functions.ProbabilityDensityFunction import \
    PDF_METADATA_ITEMS
from riser.probability_functions import value_arrays


#################### RESAMPLING/INTERPOLATION ####################
def interpolate_pdf(pdf:PDF, x:np.ndarray, verbose=False) -> PDF:
    """Resample a PDF along a new value array.

    Args    pdf - PDF to be resampled
            x - np.ndarray, value array along which to resample the PDF
    Returns pdf_resamp - resampled PDF
    """
    if verbose == True:
        print(f"Interpolating PDF to {len(x)}-length array")

    # Interpolate probability density values along the new value array
    px_resamp = np.interp(x, pdf.x, pdf.px, left=0, right=0)

    # Copy metadata from original PDF
    metadata = {}
    for meta_item in PDF_METADATA_ITEMS:
        # Retrieve metadata value from original PDF
        metadata[meta_item] = getattr(pdf, meta_item)

    # Instantiate new, resampled PDF
    pdf_resamp = PDF(x, px_resamp, **metadata)

    return pdf_resamp


def interpolate_pdfs(pdfs:list[PDF], verbose=False) -> list[PDF]:
    """Resample multiple PDFs along a common value array.
    First, determine the value limits and sample rate over which to resample.
    Then, resample each PDF accordingly.

    Args    pdfs - list[PDF], PDFs to resample
    Returns pdfs_resamp - list[PDF], resampled PDFs
    """
    # Determine value limits and sample rate
    (xmin, xmax, dx) = value_arrays.value_array_params_from_pdfs(
            pdfs, verbose=verbose)

    # Create value array
    x = value_arrays.create_precise_value_array(xmin, xmax, dx,
                                                verbose=verbose)

    # Resample PDFs
    pdfs_resamp = [interpolate_pdf(pdf, x) for pdf in pdfs]

    return pdfs_resamp


# end of file