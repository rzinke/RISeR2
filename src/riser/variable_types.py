# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
Because most RISeR2 functions are general and not specific to slip rates,
variable types should be suggested but not enforced except in the case
of slip rate calculations.
"""

# Public API
__all__ = [
    "SUPPORTED_VARIABLE_TYPES",
    "check_variable_type_supported",
]


# Constants
SUPPORTED_VARIABLE_TYPES = (
    "age",
    "displacement",
    "slip rate",
)


# Import modules
import warnings


#################### VARIABLE TYPE TYPE ####################
type VariableType = str


#################### PDF VARIABLE TYPE CHECKS ####################
def check_variable_type_supported(variable_type: str) -> None:
    """Check whether the specified variable type is supported.

    Parameters
    ----------
    variable_type : str
        Specified variable type.
    
    Returns
    -------
    None
    """
    # Check if variable type is supported
    if not variable_type in SUPPORTED_VARIABLE_TYPES:
        raise ValueError(f"Variable type '{variable_type}' not supported")


# end of file
