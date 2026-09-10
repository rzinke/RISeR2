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
from ... import probability_functions as PDFs


#################### TRIMMING FUNCTIONS ####################
def trim_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    verbose: bool = False,
) -> tuple[PDFs.PDF, PDFs.PDF]:
    """Trim two PDFs against each other, enforcing that pdf1 precedes pdf2.

    Theory:
    Given a known ordering constraint (pdf1 < pdf2), each variable's own
    distribution can be reshaped by conditioning on the other's CDF:

        pdf1_trimmed(x) ~ pdf1(x) . (1 - CDF_pdf2(x))
        pdf2_trimmed(x) ~ pdf2(x) . CDF_pdf1(x)

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
    """
    if verbose:
        print("Adding variables")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Trim first variable relative to second
    px1_trimmed = pdf1.px * (1 - pdf2.Px)

    # Trim second variable relative to first
    px2_trimmed = pdf2.px * pdf1.Px

    # Formulate output names
    trimmed_name1 = f"{pdf1.name} trimmed" if pdf1.name is not None else None
    trimmed_name2 = f"{pdf2.name} trimmed" if pdf2.name is not None else None

    # Format metadata for trimmed PDFs
    metadict1 = pdf1.metadata.as_dict()
    metadict1["name"] = trimmed_name1

    metadict2 = pdf2.metadata.as_dict()
    metadict2["name"] = trimmed_name2

    # Format results as PDFs
    pdf1_trimmed = PDFs.PDF(
        x=pdf1.x,
        px=px1_trimmed,
        **metadict1,
    )

    pdf2_trimmed = PDFs.PDF(
        x=pdf2.x,
        px=px2_trimmed,
        **metadict2,
    )

    return pdf1_trimmed, pdf2_trimmed


# end of file
