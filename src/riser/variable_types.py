# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
Because most RISeR2 functions are general and not specific to slip rates,
variable types should be suggested but not enforced.
"""

# Constants
SUPPORTED_VARIABLE_TYPES = (
    "age",
    "displacement",
    "slip rate"
)


# Import modules
import warnings

from riser.probability_functions import PDF


#################### PDF VARIABLE TYPE CHECKS ####################
def check_variable_type_supported(
        variable_type:str, throw_error:bool=False, verbose=False
    ) -> bool:
    """Check whether the specified variable type is supported.

    Args    variable_type - str, specified variable type
            throw_error - bool, throw an error if the variable type is not
                supported
    Returns bool, True if variable type is supported
    """
    # Check if variable type is supported
    if variable_type in SUPPORTED_VARIABLE_TYPES:
        # Confirm support
        if verbose == True:
            print(f"Variable type {variable_type} is known and supported")

        return True

    else:
        # Variable type is not supported
        if throw_error == True:
            # Throw error
            raise ValueError(f"Variable type {variable_type} not supported")

        else:
            # Warn only
            warnings.warn(
                f"Variable type {variable_type} not supported",
                stacklevel=2
            )

            return False


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