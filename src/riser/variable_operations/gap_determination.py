# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Compute the probability density of values lying between two random variables.
"""


# Public API
__all__ = [
    "compute_probability_between_variables",
]


# Import modules
import copy

import numpy as np

from .. import (
    probability_functions as PDFs,
    units,
    variable_types,
)


#################### GAP BETWEEN VARIABLES ####################
def compute_probability_between_variables(
    pdf1: PDFs.PDF, 
    pdf2: PDFs.PDF,
    name: str | None = None,
    verbose: bool = False,
) -> PDFs.PDF:
    """Compute a PDF representing the domain and probability densities of
    values between two random variables.

    Theory: The probability of a value being between two uncertain values is
    equal to the probability that a value is larger than the first value
    (P(X <= x)) and smaller than the second value (1 - P(Y <= y)):

        P(X < x < Y) = CDF_X . (1 - CDF_Y) = P(X <= x) * (1 - P(Y <= y))

    Machinery: The CDFs of the first and second PDFs are pre-computed during
    PDF instantiation. Leverage these to compute the "between-PDF".

    Parameters
    ----------
    pdf1 : PDF
        Smaller PDF.
    pdf2 : PDF
        Larger PDF.
    name : str, optional
        Name of "between" PDF.

    Returns
    -------
    gap_pdf : PDF
        PDF describing values between the two input variables.
    """
    if verbose:
        print("Computing probability of a value between two variables.")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata([pdf1, pdf2], warn=True)

    # Compute probabilities between variables
    px = pdf1.Px * (1 - pdf2.Px)

    # Form results into PDF
    gap_pdf = PDFs.PDF(
        pdf1.x,
        px,
        name=name,
        variable_type=metadata.variable_type,
        unit=metadata.unit,
        normalize_area=True,
    )

    return gap_pdf


# end of file
