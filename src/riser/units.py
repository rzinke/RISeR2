# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Constants
BASE_UNITS = ['m', 'y']


UNIT_SCALES = {
    'm': 0.001,
    'c': 0.01,
    'd': 0.1,
    'D': 10.,
    'C': 100.,
    'k': 1000.,
    'M': 1000000.,
}


# Import modules
import warnings

import numpy as np

from riser.probability_functions import PDF


#################### UNIT CHECKS ####################
def check_pdf_base_unit(pdf:PDF) -> str:
    """Check whether a PDF has a unit assigned, and the base of that unit.

    Args    pdf - PDF to check
    Returns base_unit - str
    """
    # Check PDF has unit
    if pdf.unit is None:
        raise ValueError("PDF unit is not defined")

    # Determine base unit
    base_unit = pdf.unit[0]

    # Check PDF base unit is appropriate
    if pdf.unit not in BASE_UNITS:
        raise ValueError(f"PDF must have base unit {', or '.join(BASE_UNITS)}")

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


#################### UNIT SCALING ####################
def scale_from_unit_prefix(unit:str) -> float:
    len_unit = len(unit)

    if len_unit > 2:
        raise ValueError(f"Unit {unit} not recognized")

    elif len_unit == 2:
        unit_prefix = unit[0]

        return UNIT_SCALES[unit_prefix]

    else:
        return 1.0


def scale_units(input_data:np.ndarray, input_unit:str, output_unit:str,
                verbose=False) -> np.ndarray:
    """Scale data from input units to output units.
    This is a relatively simple problem because there are only two types of
    units:

        m (meter)
        y (year)

    The only combination of these units is velocity:

        m/y

    The input unit should be split by the division operator into unit-parts.
    The unit prefixes should then be isolated.

    E.g.,
        input_data: 1000
        input_units: y
        output_units: ky
        output_data: 1
    """
    if verbose == True:
        print("Parsing and scaling unit")

    # Initialize scale factor
    scale_factor = 1.0

    # Split input and output units based on division operator
    input_unit_parts = input_unit.split("/")
    output_unit_parts = output_unit.split("/")

    # Check that in/out units have the same number of parts
    if len(input_unit_parts) != len(output_unit_parts):
        raise ValueError("Input and output units must have the same "
                         "dimensions")

    recip = 1
    for in_unit, out_unit in zip(input_unit_parts, output_unit_parts):
        # Check that unit bases are the same
        if in_unit[-1] != out_unit[-1]:
            raise ValueError("Unit bases are not the same")

        # Determine unit scales from prefixes
        scale_factor *= \
                  scale_from_unit_prefix(in_unit)**recip \
                / scale_from_unit_prefix(out_unit)**recip

        recip *= -1

    # Format final unit
    return scale_factor * input_data


# end of file