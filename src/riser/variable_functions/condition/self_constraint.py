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
import numpy as np

from ... import probability_functions as PDFs
from . import core


#################### CONSTRAINT FUNCTIONS ####################
def constrain_above(
    pdf: PDFs.PDF,
    value: float,
    name: str | None = None,
    verbose: bool = False,
) -> tuple[PDFs.PDF, float]:
    """Allow non-zero probability density only above the specified value.

    I.e., enforce all-zero probability density below the value.

    Parameters
    ----------
    pdf : PDF
        PDF to constrain.
    value : float
        Value above which all probability densities will be zero.
    name : str, optional
        Name of constrained PDF.

    Returns
    -------
    pdf_constrained : PDF
        Constrained PDF.
    area : float
        The fraction of the prior retained by the constraint.
    """
    if verbose:
        print(f"Constraining PDF above {value}")

    # Formulate default output name
    default_name = f"{pdf.name} constr" if pdf.name is not None else None

    # Create weighting array
    weight = np.ones(len(pdf))
    weight[pdf.x <= value] = 0

    # Constrain PDF
    pdf_constrained, area = core.condition(
        pdf, weight, name=name if name is not None else default_name
    )

    return pdf_constrained, area


def constrain_below(
    pdf: PDFs.PDF,
    value: float,
    name: str | None = None,
    verbose: bool = False,
) -> tuple[PDFs.PDF, float]:
    """Allow non-zero probability density only below the specified value.

    I.e., enforce all-zero probability density above the value.

    Parameters
    ----------
    pdf : PDF
        PDF to constrain.
    value : float
        Value below which all probability densities will be zero.
    name : str, optional
        Name of constrained PDF.

    Returns
    -------
    pdf_constrained : PDF
        Constrained PDF.
    area : float
        The fraction of the prior retained by the constraint.
    """
    if verbose:
        print(f"Constraining PDF below {value}")

    # Formulate default output name
    default_name = f"{pdf.name} constr" if pdf.name is not None else None

    # Create weighting array
    weight = np.ones(len(pdf))
    weight[pdf.x >= value] = 0

    # Constrain PDF
    pdf_constrained, area = core.condition(
        pdf, weight, name=name if name is not None else default_name
    )

    return pdf_constrained, area


# end of file
