# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
BASE_UNITS = ("m", "y")

UNIT_SCALES = {
    "m": 0.001,
    "c": 0.01,
    "d": 0.1,
    "D": 10.,
    "C": 100.,
    "k": 1000.,
    "M": 1000000.
}


# Import modules
import warnings
import copy

import numpy as np

from riser.probability_functions import PDF


#################### UNIT PARSING ####################
def parse_unit(unit:str, verbose=False) -> (float, str):
    """Determine the components of a unit.
    Currently only works with simple units (e.g., m, y) and not compound units
    (e.g., m/y).

    Args    unit - str, unit to parse
    Returns scale - float, unit scale
            base - str, unit base
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
            raise ValueError(f"Prefix '{prefix}' not supported")
    else:
        # Set scale
        scale = 1.0

    # Determine base unit
    base = unit[-1]

    # Check that base is valid
    check_base_unit(base)

    # Report if requested
    if verbose == True:
        print(f"Unit: {scale:E} {base}")

    return scale, base


#################### UNIT CHECKS ####################
def check_base_unit(base_unit:str) -> bool:
    """Check that the base unit is supported.
    """
    # Check PDF base unit is appropriate
    if base_unit not in BASE_UNITS:
        raise ValueError(f"PDF {pdf.name} must have base unit "
                         f"{', or '.join(BASE_UNITS)}")

    return True


#################### UNIT PRIORITIZATION ####################
def get_priority_unit(file_unit:str|None, inps_unit:str|None,
                      verbose=False) -> str:
    """If a unit is specified both in the file, and by the user, prioritize
    the unit encoded in the file.
    """
    # Check if unit is specified in both the file and user inputs
    if all([file_unit is not None,
            inps_unit is not None,
            file_unit != inps_unit]):
        # Warn user
        warnings.warn("Unit specified in file is different from "
                      "user-specified unit.", stacklevel=2)

    # Set priority unit
    if all([file_unit is None, inps_unit is not None]):
        priority_unit = copy.deepcopy(inps_unit)
    else:
        priority_unit = copy.deepcopy(file_unit)

    # Check that base unit is valid
    if priority_unit is not None:
        parse_unit(priority_unit)

    # Report if requested
    if verbose == True:
        print(f"Prioritizing file unit: {priority_unit}")

    return priority_unit


#################### UNIT SCALING ####################
def scale_values_by_units(values:float|np.ndarray, unit_in:str, unit_out:str,
        verbose=False) -> float|np.ndarray:
    """Scale values from the input unit to the output.
    Currently only works with simple units (e.g., m, y) and not compound units
    (e.g., m/y).
    """
    if verbose == True:
        print(f"Scaling from {unit_in} to {unit_out}")

    # Check if compound unit
    operators = [".", "/"]
    if any(
        [char in unit for unit in [unit_in, unit_out] for char in operators]):
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


def scale_pdf_by_units(pdf:PDF, unit_out:str, verbose=False) -> \
        float|np.ndarray:
    """Scale the values of a PDF from the input unit to the output.
    Only the values and units are changed.
    Currently only works with simple units (e.g., m, y) and not compound units
    (e.g., m/y).
    """
    # Escape if units not properly specified or scaling is not desired
    if any([pdf.unit is None, unit_out is None]):
        if pdf.unit is None:
            warnings.warn("Cannot scale PDF values with units None. "
                          "Continuing with original units.", stacklevel=2)
        return pdf

    # Scale values
    scaled_values = scale_values_by_units(pdf.x, pdf.unit, unit_out,
                                          verbose=verbose)

    # Form scaled PDF
    pdf_args = {
        "x": scaled_values,
        "px": pdf.px,
        "name": pdf.name,
        "variable_type": pdf.variable_type,
        "unit": unit_out,
    }
    scaled_pdf = PDF(**pdf_args, normalize_area=True)

    return scaled_pdf


#################### PDF UNIT CHECKS ####################
def check_pdf_base_unit(pdf:PDF, variable_type:str=None) -> str:
    """Check whether a PDF has a unit assigned, and the base of that unit.

    Args    pdf - PDF to check
    Returns base_unit - str
    """
    # Check PDF has unit
    if pdf.unit is None:
        raise ValueError(f"PDF {pdf.name} unit is not defined")

    # Determine base unit
    _, base_unit = parse_unit(pdf.unit)

    # Check PDF base unit is appropriate
    check_base_unit(base_unit)

    # Check against variable type
    if all([variable_type == "age", base_unit != "y"]):
        raise ValueError("Unit for variable type age is 'y'")

    if all([variable_type == "displacement", base_unit != "m"]):
        raise ValueError("Unit for variable type displacement is 'm'")

    return base_unit


def check_same_pdf_units(pdfs:list[PDF]) -> str|None:
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


# end of file