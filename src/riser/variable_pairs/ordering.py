# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

"""
Check that dated markers are ordered from youngest to oldest.
"""

# Public API
__all__ = [
    "REVERSAL_PROBABILITY_LIMIT",
    "compute_reversal_probability",
    "find_ordering_violations",
    "check_marker_order",
]


# Import modules
from collections.abc import Mapping
from itertools import pairwise

from .. import integration, probability_functions as PDFs
from .dated_marker import DatedMarker


# Probability above which a marker is considered decisively out of order.
# Markers whose PDFs overlap are expected, and are handled by conditioning
# the variables on the known order, so this is deliberately high.
REVERSAL_PROBABILITY_LIMIT = 0.99


#################### MARKER ORDER ####################
def compute_reversal_probability(pdf: PDFs.PDF, ref_pdf: PDFs.PDF) -> float:
    """Compute the probability that a variable is smaller than a reference
    variable, assuming the two are independent.

    P(X < X_ref) = integral(f(x) * (1 - F_ref(x)) dx)

    Parameters
    ----------
    pdf : PDF
        Variable expected to be larger than the reference.
    ref_pdf : PDF
        Reference variable expected to be smaller.

    Returns
    -------
    float
        Probability that `pdf` is smaller than `ref_pdf`.
    """
    return float(
        integration.integrate(
            x=pdf.x,
            px=pdf.px * (1.0 - ref_pdf.cdf_at_value(pdf.x)),
        )
    )


def find_ordering_violations(
    markers: Mapping[str, DatedMarker],
    limit: float = REVERSAL_PROBABILITY_LIMIT,
) -> list[str]:
    """Find adjacent markers that are probably not ordered from youngest/least
    displaced to oldest/most displaced.

    Overlapping markers are not violations. A violation is reported only if
    the probability that a marker is younger (or less displaced) than the
    previous marker exceeds `limit`.

    Parameters
    ----------
    markers : Mapping[str, DatedMarker]
        Dated markers, expected from youngest to oldest.
    limit : float
        Reversal probability above which a violation is reported.

    Returns
    -------
    violations : list[str]
        Description of each violation, e.g.,
        "Marker 'A' appears to be younger than 'B' (probability 1.00).".
        Empty if no markers are decisively out of order.
    """
    violations = []

    for ref_marker, marker in pairwise(markers.values()):
        # Check that marker is older than previous
        p_age = compute_reversal_probability(marker.age, ref_marker.age)
        if p_age > limit:
            violations.append(
                f"Marker '{marker.name}' appears to be younger "
                f"than '{ref_marker.name}' (probability {p_age:.2f})."
            )

        # Check that marker is more displaced than previous
        p_disp = compute_reversal_probability(
            marker.displacement, ref_marker.displacement
        )
        if p_disp > limit:
            violations.append(
                f"Marker '{marker.name}' appears to be less displaced "
                f"than '{ref_marker.name}' (probability {p_disp:.2f})."
            )

    return violations


def check_marker_order(
    markers: Mapping[str, DatedMarker],
    limit: float = REVERSAL_PROBABILITY_LIMIT,
) -> None:
    """Raise an error if markers are decisively out of order.

    Markers with overlapping ages or displacements are allowed, because the
    known order can be used to update the variables. Only markers that are
    almost certainly in the wrong order raise an error.

    Parameters
    ----------
    markers : Mapping[str, DatedMarker]
        Dated markers, expected from youngest to oldest.
    limit : float
        Reversal probability above which an error is raised.

    Raises
    ------
    ValueError
        If any adjacent markers are decisively out of order.
    """
    violations = find_ordering_violations(markers, limit=limit)

    if violations:
        raise ValueError(
            "Markers must be listed from youngest/least displaced to "
            "oldest/most displaced. " + " ".join(violations)
        )


# end of file
