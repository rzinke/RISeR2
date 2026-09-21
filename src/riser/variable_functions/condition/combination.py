# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
Functions to fuse multiple independent observations of a single event.
"""


# Public API
__all__ = [
    "combine_variables",
]


# Import modules
from ... import probability_functions as PDFs
from . import core


#################### COMBINATION FUNCTIONS ####################
def combine_variables(
    pdfs: list[PDFs.PDF],
    name: str | None = None,
    verbose: bool = False,
) -> tuple[PDFs.PDF, float]:
    """Compute the joint probability mass function of two or more discrete
    random variables.

        fcomb(x) = f1(x) . f2(x) . ... fn(x) = product(fi(x))

    The area of the combined PDF indicates how similar or compatible the PDFs
    are.

    Parameters
    ----------
    pdfs : list[PDF]
        List of PDFs to combine.
    name : str, optional
        Descriptive name of combined PDF.

    Returns
    -------
    pdf_combined : PDF
        Combined pdf.
    area : float
        Likelihood of the combined estimate, reflecting how compatible the 
        independent estimates are with one another.
    """
    if verbose:
        print(f"Combining {len(pdfs)} PDFs")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling(pdfs)

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf.metadata for pdf in pdfs])

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata(
        [pdf.metadata for pdf in pdfs], name=name
    )

    # Initialize the combined PDF based on the first PDF in the list
    pdf_combined = PDFs.PDF(
        x=pdfs[0].x,
        px=pdfs[0].px,
        **metadata.as_dict(),
    )

    # Initialize posterior kernel
    kernel = PDFs.weight_functions.WeightFunction.from_pdf(pdfs[0])

    # Loop through subsequent PDFs
    for pdf in pdfs[1:]:
        # Condition the combined weight function
        kernel = core.weigh(kernel, PDFs.weight_functions.WeightFunction.from_pdf(pdf))

    # Convert the combined weight function to a PDF
    pdf_combined = kernel.normalize(**metadata.as_dict())
    area = kernel.area

    return pdf_combined, area


# end of file
