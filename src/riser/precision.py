# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Constants
RISER_PRECISION = 10  # decimals


# Import modules
import warnings

import numpy as np


#################### ROUNDING ####################
def check_precision(x: float) -> float:
    """Check that value is above precision limit.
    """
    if x <= 10**-RISER_PRECISION:
        warnings.warn(
            "Number is less than optimal precision of the RISeR library"
        )


def fix_precision(x: float) -> float:
    """Round to a tiny digit to compensate for machine error.
    """
    return np.round(x, RISER_PRECISION)


# end of file