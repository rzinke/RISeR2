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
    *,
    # PDF metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
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
        Name of summed PDF.
    variable_type : str, optional
        Variable type of summed PDF.
    unit : str, optional
        Unit of summed PDF.

    Returns
    -------
    combined_pdf : PDF
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
    common_metadata = PDFs.metadata.get_common_metadata(
        [pdf.metadata for pdf in pdfs], name=name,
    )

    # Format metadata
    metadata_dict = common_metadata.as_dict()
    if variable_type is not None:
        metadata_dict["variable_type"] = variable_type
    if unit is not None:
        metadata_dict["unit"] = unit

    # Initialize posterior kernel
    kernel = PDFs.weight_functions.WeightFunction.from_pdf(pdfs[0])

    # Loop through subsequent PDFs
    for pdf in pdfs[1:]:
        # Condition the combined weight function
        kernel = core.weigh(kernel, PDFs.weight_functions.WeightFunction.from_pdf(pdf))

    # Convert the combined weight function to a PDF
    combined_pdf = kernel.normalize(**metadata_dict)
    area = kernel.area

    return combined_pdf, area


# end of file
