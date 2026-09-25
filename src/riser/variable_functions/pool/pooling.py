# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
Pooling is the process of creating a mixture distribution that combines
several different events into a single, aggregate distribution.
"""


# Public API
__all__ = [
    "pool_variables",
]


# Import modules
import copy

from ... import probability_functions as PDFs


#################### POOLING FUNCTIONS ####################
def pool_variables(
    pdfs: list[PDFs.PDF],
    *,
    # PDF metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> PDFs.PDF:
    """Combine two or more probability mass.

    Combine distributions by summing them pointwise

    fpool(x) = f1(x) + f2(x) + ... + fn(x) = sum(fi(x))

    and normalizing the area.

    Note that "merging" has no formal definition in the context of probability
    theory.
    This is similar to the OxCal sum function, and should not be confused with
    either compute_joint_pdf (which combines PDFs by multiplying them element-
    wise) or add_variables (which computes the sum of two independent random
    variables).
    OxCal provides a note:
    '... the 95% range for a Sum distribution give an estimate for the period
    in which 95% of the events took place not the period in which one can be
    95% sure all of the events took place.'

    Parameters
    ----------
    pdfs : list[PDF]
        List of PDFs to pool.
    name : str, optional
        Name of pooled PDF.
    variable_type : str, optional
        Variable type of pooled PDF.
    unit : str, optional
        Unit of pooled PDF.

    Returns
    -------
    pooled_pdf : PDF
        Pooled PDF.
    """
    if verbose:
        print(f"Pooling {len(pdfs)} PDFs")

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

    # Initialize probability density array
    x = copy.deepcopy(pdfs[0].x)
    px = copy.deepcopy(pdfs[0].px)

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px += pdf.px

    # Form results into PDF
    pooled_pdf = PDFs.PDF(x=x, px=px, **metadata_dict)

    return pooled_pdf


# end of file
