# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
These functions combine random variables in different ways.
"""


# Public API
__all__ = [
    "combine_variables",
    "merge_variables",
]


# Import modules
import copy

from .. import (
    probability_functions as PDFs,
    units,
    variable_types,
)


#################### RANDOM VARIABLE COMBINATION ####################
def combine_variables(pdfs: list[PDFs.PDF], verbose: bool = False) -> PDFs.PDF:
    """Compute the joint probability mass function of two or more discrete
    random variables.
    Note: Treating the PDFs as discrete greatly simplifies the calculations.

    f_X,Y(x,y) = f_X(x) * f_Y(y)

    This is similar to OxCal R_Combine.

    Parameters
    ----------
    pdfs : list[PDF]
        List of PDFs to combine.

    Returns
    -------
    joint_pdf : PDF
        Joint pdf.
    """
    if verbose:
        print(f"Combining {len(pdfs)} PDFs")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling(pdfs)

    # Check variable types
    variable_type = variable_types.check_same_pdf_variable_types(pdfs)

    # Check units
    unit = units.check_same_pdf_units(pdfs)

    # Base PDF
    px = copy.deepcopy(pdfs[0].px)

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px *= pdf.px

    # Form results into PDF
    joint_pdf = PDFs.PDF(pdfs[0].x, px, normalize_area=True, unit=unit)

    return joint_pdf


def merge_variables(pdfs: list[PDFs.PDF], verbose: bool = False) -> PDFs.PDF:
    """Combine two or more probability mass.

    Combine distributions by summing them pointwise

    p = f_X(x) + f_Y(y)

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
        List of PDFs to merge.

    Returns
    -------
    merged_pdf : PDF
        Merged PDF.
    """
    if verbose:
        print(f"Merging {len(pdfs)} PDFs")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling(pdfs)

    # Check variable types
    variable_type = variable_types.check_same_pdf_variable_types(pdfs)

    # Check units
    unit = units.check_same_pdf_units(pdfs)

    # Base PDF
    px = copy.deepcopy(pdfs[0].px)

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px += pdf.px

    # Form results into PDF
    merged_pdf = PDFs.PDF(pdfs[0].x, px, normalize_area=True, unit=unit)

    return merged_pdf


# end of file
