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
import copy

from ... import probability_functions as PDFs


#################### COMBINATION FUNCTIONS ####################
def combine_variables(
    pdfs: list[PDFs.PDF],
    name: str | None = None,
    verbose: bool = False,
) -> PDFs.PDF:
    """Compute the joint probability mass function of two or more discrete
    random variables.
    Note: Treating the PDFs as discrete greatly simplifies the calculations.

    fcomb(x) = f1(x) * f2(x) * ... fn(x) = product(fi(x))

    Parameters
    ----------
    pdfs : list[PDF]
        List of PDFs to combine.
    name : str, optional
        Descriptive name of combined PDF.

    Returns
    -------
    combined_pdf : PDF
        Combined pdf.
    """
    if verbose:
        print(f"Combining {len(pdfs)} PDFs")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling(pdfs)

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata(
        [pdf.metadata for pdf in pdfs], name=name, warn=True
    )

    # Base PDF
    px = copy.deepcopy(pdfs[0].px)

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px *= pdf.px

    # Form results into PDF
    combined_pdf = PDFs.PDF(
        x=pdfs[0].x,
        px=px,
        **metadata.as_dict(),
    )

    return combined_pdf


# end of file
