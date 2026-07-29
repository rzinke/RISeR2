# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Public API
__all__ = [
    "BASE_UNITS",
    "UNIT_SCALES",
    "parse_unit",
    "check_base_unit",
    "scale_by_units",
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


#################### UNIT PARSING ####################
def parse_unit(
    unit: str | None, verbose: bool = False
) -> tuple[float, str] | tuple[None, None]:
    """Determine the components of a unit.

    Currently only works with simple units (e.g., m, y) and not compound units
    (e.g., m/y).

    Parameters
    ----------
    unit : str or None
        Unit to parse.

    Returns
    -------
    scale : float
        Unit scale.
    base : str
        Unit base.
    """
    if unit is None:
        if verbose:
            print("Unit is 'None'")

        return None, None

    else:
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
                raise ValueError(f"Prefix '{prefix}' not supported")
        else:
            # Set scale
            scale = 1.0

        # Determine base unit
        base = unit[-1]

        # Check that base is valid
        check_base_unit(base)

        # Report if requested
        if verbose:
            print(f"Unit: {scale:E} {base}")

        return scale, base


#################### UNIT CHECKS ####################
def check_base_unit(base_unit: str):
    """Check that the base unit is supported.

    Parameters
    ----------
    base_unit : str
        Base unit.

    Returns
    -------
    bool
        True if base unit is supported.
    """
    # Check base unit is appropriate
    if base_unit not in BASE_UNITS:
        raise ValueError(
            f"Base unit '{base_unit}' not supported. "
            f"Use one of {', '.join(BASE_UNITS)}"
        )


#################### UNIT SCALING ####################
def scale_by_units(
    values: float | np.ndarray,
    unit_in: str,
    unit_out: str,
    verbose: bool = False,
) -> float | np.ndarray:
    """Scale values from the input unit to the output.

    Currently only works with simple units (e.g., m, y) and not compound units
    (e.g., m/y).

    Parameters
    ----------
    values : float or np.ndarray
        Values to scale by change in output units.
    unit_in : str
        Original unit.
    unit_out : str
        Output unit.

    Returns
    -------
    scaled_values : float or np.ndarray
        Values scaled by the change in output units.
    """
    if verbose:
        print(f"Scaling from {unit_in} to {unit_out}")

    # Check if scaling is appropriate
    if unit_in is None:
        raise ValueError(
            "Original unit must be defined for scaling. Got 'None'."
        )

    if unit_out is None:
        raise ValueError(
            "Output unit must be defined for scaling. Got 'None'."
        )

    # Check if compound unit
    operators = [".", "/"]
    if any(
        [char in unit for unit in [unit_in, unit_out] for char in operators]
    ):
        raise ValueError("Compound units not currently supported")

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
