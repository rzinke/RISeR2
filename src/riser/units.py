# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Unit parsing and scaling for RISeR2.

Convention for functions in this module that accept a unit string:

- Functions that COMPUTE a result from a unit (parse_unit,
  scale_values_by_units) require a real unit and raise if given `None`.
  There is no meaningful default scale or base to fall back on, so
  continuing silently would risk propagating an incorrect value rather
  than failing clearly.
- Functions that VALIDATE a unit (check_base_unit_supported) accept
  `None` but warn, since silently skipping validation would hide the
  exact condition the check exists to catch.
- Functions that DESCRIBE whether an operation is possible
  (determine_if_scaling_appropriate, in probability_functions.scaling)
  accept `None` and return a plain False/no-op, since "no unit means
  no scaling" is an expected, legitimate outcome for a caller to receive.
"""

# Public API
__all__ = [
    "BASE_UNITS",
    "UNIT_SCALES",
    "parse_unit",
    "check_base_unit_supported",
    "scale_values_by_units",
]


# Constants
BASE_UNITS = ("m", "y")

UNIT_SCALES = {
    "m": 0.001,
    "c": 0.01,
    "d": 0.1,
    "D": 10.,
    "C": 100.,
    "k": 1000.,
    "M": 1000000.,
}


# Import modules
import warnings
import copy

import numpy as np


#################### UNIT TYPE ####################
type Unit = str


#################### UNIT CHECKS ####################
def check_base_unit_supported(base_unit: str | None) -> None:
    """Check that the base unit is supported.

    Warns if base unit is `None`.

    Parameters
    ----------
    base_unit : str or None
        Base unit.

    Returns
    -------
    None
    """
    # Warn if base unit is None
    if base_unit is None:
        warnings.warn(
            "Base unit 'None' is not supported for unit-based operations"
        )

        return

    # Check base unit is supported
    if base_unit not in BASE_UNITS:
        raise ValueError(
            f"Base unit '{base_unit}' not supported. "
            f"Use one of {', '.join(BASE_UNITS)}"
        )


#################### UNIT PARSING ####################
def parse_unit(
    unit: str, verbose: bool = False
) -> tuple[float, str]:
    """Determine the components of a unit.

    Currently only works with simple units (e.g., m, y) and not compound units
    (e.g., m/y).

    Parameters
    ----------
    unit : str
        Unit to parse (cannot be `None`).

    Returns
    -------
    scale : float
        Unit scale.
    base : str
        Unit base.
    """
    # Check if unit as an exponent
    if unit[-1].isdigit():
        raise ValueError("Exponents are not currently supported")

    # Check unit formatting based on length
    if len(unit) > 2:
        raise ValueError("Unit not recognized")

    # Determine unit scale
    if len(unit) == 2:
        # Unit prefix
        prefix = unit[0]

        # Determine scale from prefix
        scale = UNIT_SCALES.get(prefix)

        # Check prefix is valid
        if scale is None:
            raise ValueError(f"Unit prefix '{prefix}' not supported")
    else:
        # Set scale
        scale = 1.0

    # Determine base unit
    base = unit[-1]

    # Check that base is valid
    check_base_unit_supported(base)

    # Report if requested
    if verbose:
        print(f"Unit: {scale:E} {base}")

    return scale, base


#################### UNIT SCALING ####################
def scale_values_by_units[Values: (float, np.ndarray)](
    values: Values,
    unit_in: str | None,
    unit_out: str | None,
    verbose: bool = False,
) -> Values:
    """Scale values from the input unit to the output.

    Currently only works with simple units (e.g., m, y) and not compound units
    (e.g., m/y).

    Parameters
    ----------
    values : float or np.ndarray
        Values to scale by change in output units.
    unit_in : str or None
        Original unit.
    unit_out : str or None
        Output unit.

    Returns
    -------
    scaled_values : float or np.ndarray
        Values scaled by the change in output units.
    """
    if verbose:
        print(f"Scaling from {unit_in} to {unit_out}")

    # Check if scaling is appropriate
    if unit_in is None or unit_out is None:
        raise ValueError(
            f"Neither input unit ({unit_in}) nor output unit ({unit_out}) "
            f"can be 'None'"
        )

    # Check if compound unit
    operators = [".", "/"]
    if any(
        [char in unit for unit in [unit_in, unit_out] for char in operators]
    ):
        raise ValueError(
            f"Compound units not currently supported, "
            f"got input unit '{unit_in}' and output unit '{unit_out}'"
        )

    # Parse input and output units
    scale_in, base_in = parse_unit(unit_in)
    scale_out, base_out = parse_unit(unit_out)

    # Check whether input and output units are compatible (same base)
    if base_out != base_in:
        raise ValueError(f"Units do not match (in {base_in}, out {base_out})")

    # Determine scale
    scale_factor = scale_in / scale_out

    return scale_factor * values


# end of file
