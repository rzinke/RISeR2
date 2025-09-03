# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
Because most RISeR2 functions are general and not specific to slip rates,
variable types should be suggested but not enforced.
"""

# Constants
SLIPRATE_VARIABLE_TYPES = ["age", "displacement", "slip rate"]


# Import modules
from riser.probability_functions import PDF


#################### PDF VARIABLE TYPE CHECKS ####################
def check_same_pdf_variable_types(pdfs:list[PDF]) -> str|None:
    """Check that the variable types are the same among a series of PDFs.

    Args    pdfs - list[PDF], PDFs to check
    Returns variable type - str|None, common variable type, if variable type
                is common to all PDFs
    """
    # Establish initial variable type
    variable_type = pdfs[0].variable_type

    # Loop through all PDFs
    for pdf in pdfs[1:]:
        # Check PDF variable type against initial
        if pdf.variable_type != variable_type:
            # Nullify variable type
            variable_type = None

    return variable_type


# end of file