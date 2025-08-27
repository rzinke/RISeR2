# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Constants


# Import modules
import warnings

from riser.probability_functions import PDF


#################### UNIT CHECKS ####################
def check_units(pdfs:list[PDF]) -> str|None:
    """Check that the units are the same among a series of PDFs.

    Args    pdfs - list[PDF], PDFs to check
    Returns unit - str|None, common unit, if unit is common to all PDFs
    """
    # Establish initial unit
    unit = pdfs[0].unit

    # Loop through all PDFs
    for pdf in pdfs[1:]:
        # Check PDF unit against initial
        if pdf.unit != unit:
            # Nullify unit
            unit = None

    return unit


#################### UNIT SCALING ####################


# end of file