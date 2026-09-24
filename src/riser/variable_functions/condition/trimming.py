# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
Condition variables based on a prior ordering relationships.
"""


# Public API
__all__ = [
    "trim_variables",
]


# Import modules
import copy

from ... import probability_functions as PDFs
from . import core


#################### TRIMMING FUNCTIONS ####################
def trim_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    name1: str | None = None,
    name2: str | None = None,
    verbose: bool = False,
) -> tuple[PDFs.PDF, PDFs.PDF, float]:
    """Trim two PDFs against each other, enforcing that pdf1 precedes pdf2.

    Theory:
    Given a known ordering constraint (pdf1 < pdf2), each variable's own
    distribution can be reshaped by conditioning on the other's CDF:

        pdf1_trimmed(x) ~ pdf1(x) . (1 - CDF2(x))
        pdf2_trimmed(x) ~ pdf2(x) . CDF1(x)

    The areas of the two trimming results will be the same, because
    
        area1 = integral(pdf1(x) . (1 - CDF2(x)) dx) = P(X1 < X2)
        area2 = integral(pdf2(x) . CDF1(x) dx) = P(X2 > X1)

    which are equivalent statements.

    Parameters
    ----------
    pdf1 : PDF
        Variable constrained to precede pdf2.
    pdf2 : PDF
        Variable constrained to follow pdf1.

    Returns
    -------
    pdf1_trimmed : PDF
        pdf1, reshaped to reflect that it must precede pdf2.
    pdf2_trimmed : PDF
        pdf2, reshaped to reflect that it must follow pdf1.
    area : float
        P(pdf1 < pdf2), or equivalently P(pdf2 > pdf1).
    """
    if verbose:
        print("Trimming variables")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Get common metadata
    metadata_dict = PDFs.metadata.get_common_metadata([pdf1.metadata, pdf2.metadata]).as_dict()
    metadata_dict1 = copy.copy(metadata_dict)
    metadata_dict2 = copy.copy(metadata_dict)

    # Formulate trimmed PDF names
    default_name1 = f"{pdf1.name} trimmed" if pdf1.name is not None else None
    default_name2 = f"{pdf2.name} trimmed" if pdf2.name is not None else None

    metadata_dict1["name"] = name1 if name1 is not None else default_name1
    metadata_dict2["name"] = name2 if name2 is not None else default_name2
    
    # Trim first variable relative to second
    weight_larger = PDFs.weight_functions.WeightFunction(
        x=pdf2.x, wx=(1 - pdf2.Px)
    )
    pdf1_trimmed, area = core.condition(
        pdf1,
        weight_larger,
        **metadata_dict1,
    )

    # Trim second variable relative to first
    weight_smaller = PDFs.weight_functions.WeightFunction(
        x=pdf1.x, wx=pdf1.Px
    )
    pdf2_trimmed, _ = core.condition(
        pdf2,
        weight_smaller,
        **metadata_dict2,
    )

    return pdf1_trimmed, pdf2_trimmed, area


# end of file
