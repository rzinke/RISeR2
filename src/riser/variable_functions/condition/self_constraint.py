# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
Impose a logical constraint that a variable is greater or less than some
value.
"""

# Public API
__all__ = [
    "constrain_above",
    "constrain_below",
]


# Import modules
from ... import probability_functions as PDFs
from . import core


#################### CONSTRAINT FUNCTIONS ####################
def constrain_above(
    pdf: PDFs.PDF,
    value: float,
    *,
    crop: bool = False,
    # PDF metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> tuple[PDFs.PDF, float]:
    """Allow non-zero probability density only above the specified value.

    I.e., enforce all-zero probability density below and at the value.

    Parameters
    ----------
    pdf : PDF
        PDF to constrain.
    value : float
        Value above which all probability densities will be zero.
    crop : bool, optional
        Crop the output PDF to the range above the specified value.
    name : str, optional
        Name of constrained PDF.
    variable_type : str, optional
        Variable type of constrained PDF.
    unit : str, optional
        Unit of constrained PDF.

    Returns
    -------
    constrained_pdf : PDF
        Constrained PDF.
    area : float
        The fraction of the prior retained by the constraint.
    """
    if verbose:
        print(f"Constraining PDF above {value}")

    # Retrieve metadata from PDF
    metadata_dict = pdf.metadata.as_dict()

    # Apply user-specified metadata values
    if name is not None:
        metadata_dict["name"] = name

    if variable_type is not None:
        metadata_dict["variable_type"] = variable_type

    if unit is not None:
        metadata_dict["unit"] = unit

    # Create weighting array
    weight = PDFs.weight_functions.zero_where(pdf.x, pdf.x <= value)

    # Constrain PDF
    constrained_pdf, area = core.condition(pdf, weight, **metadata_dict)

    # Crop constrained PDF
    if crop:
        # Overwrite constrained PDF with cropped version
        constrained_pdf = PDFs.interpolation.interpolate_pdf(
            pdf=constrained_pdf,
            x=constrained_pdf.x[constrained_pdf.x > value],
        )

    return constrained_pdf, area


def constrain_below(
    pdf: PDFs.PDF,
    value: float,
    *,
    crop: bool = False,
    # PDF metadata
    name: str | None = None,
    variable_type: str | None = None,
    unit: str | None = None,
    # Misc
    verbose: bool = False,
) -> tuple[PDFs.PDF, float]:
    """Allow non-zero probability density only below the specified value.

    I.e., enforce all-zero probability density below and at the value.

    Parameters
    ----------
    pdf : PDF
        PDF to constrain.
    value : float
        Value below which all probability densities will be zero.
    crop : bool, optional
        Crop the output PDF to the range below the specified value.
    name : str, optional
        Name of constrained PDF.
    variable_type : str, optional
        Variable type of constrained PDF.
    unit : str, optional
        Unit of constrained PDF.

    Returns
    -------
    constrained_pdf : PDF
        Constrained PDF.
    area : float
        The fraction of the prior retained by the constraint.
    """
    if verbose:
        print(f"Constraining PDF below {value}")

    # Retrieve metadata from PDF
    metadata_dict = pdf.metadata.as_dict()

    # Apply user-specified metadata values
    if name is not None:
        metadata_dict["name"] = name

    if variable_type is not None:
        metadata_dict["variable_type"] = variable_type

    if unit is not None:
        metadata_dict["unit"] = unit

    # Create weighting array
    weight = PDFs.weight_functions.zero_where(pdf.x, pdf.x >= value)

    # Constrain PDF
    constrained_pdf, area = core.condition(pdf, weight, **metadata_dict)

    # Crop constrained PDF
    if crop:
        # Overwrite constrained PDF with cropped version
        constrained_pdf = PDFs.interpolation.interpolate_pdf(
            pdf=constrained_pdf,
            x=constrained_pdf.x[constrained_pdf.x < value],
        )

    return constrained_pdf, area


# end of file
