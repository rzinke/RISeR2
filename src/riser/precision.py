# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "check_precision",
    "fix_precision",
]


# Constants
RISER_PRECISION = 10  # decimals


# Import modules
import warnings

import numpy as np


#################### ROUNDING ####################
def check_precision(x: float) -> None:
    """Check that value is above precision limit.

    Parameters
    ----------
    x : float
        Value for which to check precision.
    """
    if x <= 10 ** -RISER_PRECISION:
        warnings.warn(
            "Number is less than optimal precision of the RISeR library"
        )


def fix_precision(x: float) -> float:
    """Round to a tiny digit to compensate for machine error.

    Parameters
    ----------
    x : float
        Value for which to fix precision.

    Returns
    -------
    float
        Value with fixed precision.
    """
    return np.round(x, RISER_PRECISION)


# end of file
