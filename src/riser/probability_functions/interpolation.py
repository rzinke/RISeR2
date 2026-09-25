# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.


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

# Public API
__all__ = [
    "interpolate_pdf",
    "interpolate_pdfs",
]


# Import modules
import numpy as np

from . import value_arrays
from .metadata import METADATA_ITEMS
from .probability_density_function import ProbabilityDensityFunction as PDF


#################### RESAMPLING/INTERPOLATION ####################
def interpolate_pdf(
    pdf: PDF,
    x: np.ndarray,
    *,
    # PDF metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> PDF:
    """Resample a PDF along a new value array.

    Parameters
    ----------
    pdf : PDF
        PDF to be resampled.
    x : np.ndarray
        Value array along which to resample the PDF.
    name : str, optional
        Name of interpolated PDF.
    variable_type : str, optional
        Variable type of interpolated PDF.
    unit : str, optional
        Unit of interpolated PDF.

    Returns
    -------
    pdf_resamp : PDF
        Resampled PDF.
    """
    if verbose:
        print(f"Interpolating PDF to {len(x)}-length array")

    # Interpolate probability density values along the new value array
    px_resamp = np.interp(x, pdf.x, pdf.px, left=0, right=0)

    # Format metadata
    metadata_dict = pdf.metadata.as_dict()

    if name is not None:
        metadata_dict["name"] = name

    if variable_type is not None:
        metadata_dict["variable_type"] = variable_type

    if unit is not None:
        metadata_dict["unit"] = unit

    # Instantiate new, resampled PDF
    pdf_resamp = PDF(x=x, px=px_resamp, **metadata_dict)

    return pdf_resamp


def interpolate_pdfs(pdfs: list[PDF], verbose: bool = False) -> list[PDF]:
    """Resample multiple PDFs along a common value array.
    First, determine the value limits and sample rate over which to resample.
    Then, resample each PDF accordingly.

    Parameters
    ----------
    pdfs : list[PDF]
        PDFs to resample.

    Returns
    -------
    pdfs_resamp : list[PDF]
        Resampled PDFs.
    """
    # Determine value limits and sample rate
    xmin, xmax, dx = value_arrays.value_array_params_from_pdfs(
        pdfs, verbose=verbose
    )

    # Create value array
    x = value_arrays.precise_array(xmin, xmax, dx, verbose=verbose)

    # Resample PDFs
    pdfs_resamp = [interpolate_pdf(pdf, x) for pdf in pdfs]

    return pdfs_resamp


# end of file
