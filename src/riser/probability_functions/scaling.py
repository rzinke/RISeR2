# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Unit-based operations for PDFs.
"""

# Public API
__all__ = [
    "scale_pdf_by_units",
]


# Import modules
import warnings
import copy

from .. import units
from .probability_density_function import ProbabilityDensityFunction as PDF


#################### UNIT SCALING ####################
def determine_scaling_appropriate(
    pdf: PDF,
    unit_out: str,
    verbose: bool = False,
) -> bool:
    """Determine if scaling a PDF is appropriate based on units.

    For scaling to be appropriate, output unit must be a non-trivial scalar of
    the PDF unit.

    Parameters
    ----------
    pdf : PDF
        PDF to scale by change in output units.
    unit_out : str
        Unit by which to scale the PDF.

    Returns
    -------
    bool
        True if scaling is appropriate otherwise False.
    """
    # Check if unit specified for PDF
    pdf_unit_defined = (pdf.unit is not None)

    # Check if output unit specified
    out_unit_defined = (unit_out is not None)

    # Flag warning if user expects scaling for output but no input unit
    if not pdf_unit_defined and out_unit_defined:
        warnings.warn(
            "PDF unit must be defined for scaling (got 'None') "
            "but output unit specified as '{unit_out}'"
        )

    # Parse unit scale and base
    pdf_scale, pdf_base = units.parse_unit(pdf.unit)
    out_scale, out_base = units.parse_unit(unit_out)

    # Check base units are same
    same_base = (out_base == pdf_base)

    # Check scales are different
    output_scalar = (out_scale != pdf_scale)

    # Check all criteria met
    if all(
        [
            pdf_unit_defined,
            out_unit_defined,
            same_base,
            output_scalar,
        ]
    ):
        if verbose:
            print(
                f"Scaling PDF input unit ({pdf.unit}) "
                f"to output unit ({unit_out})"
            )
        return True
    else:
        return False


def scale_pdf_by_units(
    pdf: PDF,
    unit_out: str,
    verbose: bool = False,
) -> PDF:
    """Scale the values of a PDF from the input unit to the output.

    Only the values and units are changed.

    If scaling is determined to be inappropriate, the original PDF will
    be returned.

    Parameters
    ----------
    pdf : PDF
        PDF to scale by change in output units.
    unit_out : str
        Unit by which to scale the PDF.

    Returns
    -------
    scaled_pdf : PDF
        PDF with value axis scaled by the change in output units.
    """
    # Check if scaling is appropriate
    if determine_scaling_appropriate(
        pdf=pdf,
        unit_out=unit_out,
        verbose=verbose,
    ):
        # Scale values
        scaled_values = units.scale_values_by_units(
            values=pdf.x,
            unit_in=pdf.unit,
            unit_out=unit_out,
            verbose=verbose,
        )

        # Form scaled PDF
        scaled_pdf = PDF(
            x=scaled_values,
            px=pdf.px,
            normalize_area=True,
            name=pdf.name,
            variable_type=pdf.variable_type,
            unit=unit_out,
        )

        return scaled_pdf

    else:
        return copy.deepcopy(pdf)


# end of file
