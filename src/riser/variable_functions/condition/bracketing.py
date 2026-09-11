# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
Functions for bracketing an unknown event by computing the probability density
of a spread of values lying between two observations.
"""


# Public API
__all__ = [
    "infer_bracketed",
]


# Import modules
from ... import probability_functions as PDFs
from . import core


#################### BRACKETING FUNCTIONS ####################
def infer_bracketed(
    pdf1: PDFs.PDF, 
    pdf2: PDFs.PDF,
    name: str | None = None,
    verbose: bool = False,
) -> tuple[PDFs.PDF, float]:
    """Compute a PDF representing the domain and probability densities of
    values between two random variables.

    Theory: The probability of a value being between two uncertain values is
    equal to the probability that a value is less than or equal to the first
    value (P(X1 <= x)) and smaller than the second value (1 - P(X2 <= x)):

        P(X1 < x < X2) = CDF_X1 . (1 - CDF_X2) = P(X1 <= x) . (1 - P(X2 <= x))

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
    pdf_bracketed : PDF
        PDF describing values between the two input variables.
    area : float
        Area of the conditioned distribution, before scaling.
    """
    if verbose:
        print("Computing probability density of values between two variables.")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata(
        [pdf1.metadata, pdf2.metadata]
    )

    # Create a ones-distribution representing a shapeless prior
    prior = PDFs.PDF(
        x=pdf1.x,
        px=core.flat_weight(pdf1.x),
        **metadata.as_dict(),
    )

    # Compute the weighting distribution based on the shape of the priors
    weight = pdf1.Px * (1 - pdf2.Px)

    # Apply Bayesian condition
    pdf_bracketed, area = core.condition(prior, weight, name=name)

    return pdf_bracketed, area


# end of file
