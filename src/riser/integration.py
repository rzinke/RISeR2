# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
Define a method for integrating a probability density function that is
uniform across this library.
This can conveniently be changed as Python, NumPy, or SciPy update.
"""

# Import modules
import numpy as np
import scipy as sp


#################### INTEGRATION METHODS ####################
def integrate(
    *,
    x: np.ndarray,
    px: np.ndarray,
) -> float:
    """Compute the probability mass over the defined domain.

    Parameters
    ----------
    x : np.ndarray
        PDF value array.
    px : np.ndarray
        PDF probability density array.

    Returns
    -------
    float
        Integrated value.
    """
    return sp.integrate.trapezoid(px, x)


def integrate_cumulative(
    *,
    x: np.ndarray,
    px: np.ndarray,
) -> np.ndarray:
    """Compute the cumulative integral over the defined domain.

    Parameters
    ----------
    x : np.ndarray
        PDF value array.
    px : np.ndarray
        PDF probability density array.

    Returns
    -------
    np.ndarray
        Integrated values.
    """
    return sp.integrate.cumulative_trapezoid(px, x, initial=0)


# end of file
