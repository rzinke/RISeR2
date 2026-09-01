# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

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
def check_variable_type_supported(variable_type: str | None) -> None:
    """Check whether the specified variable type is supported.

    Warns if variable type is `None`.
    Silent if variable type is valid.
    Raises if variable type is not supported.

    Parameters
    ----------
    variable_type : str or None
        Specified variable type.
    
    Returns
    -------
    None
    """
    # Warn if base unit is None
    if variable_type is None:
        warnings.warn(
            f"Variable type 'None' not specified. "
            f"It is strongly suggested to specify variable type, "
            f"e.g., {', '.join(SUPPORTED_VARIABLE_TYPES)}"
        )

        return

    # Check if variable type is supported
    if not variable_type in SUPPORTED_VARIABLE_TYPES:
        raise ValueError(
            f"Variable type '{variable_type}' not supported. "
            f"Use one of {', '.join(SUPPORTED_VARIABLE_TYPES)}"
        )


# end of file
