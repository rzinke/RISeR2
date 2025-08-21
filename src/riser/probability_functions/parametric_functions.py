# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import numpy as np
import scipy as sp


#################### BOXCAR FUNCTIONS ####################
def boxcar(x:np.ndarray, xmin:float, xmax:float) -> np.ndarray:
    """ Boxcar function with unit area. """
    # Number of data points
    n = len(x)

    # Initialize probability values
    px = np.zeros(n)

    # Probability density values
    boxcar_ndx = (x > xmin) & (x < xmax)
    px[boxcar_ndx] = 1.0

    # Normalize area
    px /= np.sum(px)

    return px


#################### TRIANGULAR FUNCTIONS ####################
def triangular(
        x:np.ndarray, xmin:float, xmode:float, xmax:float) -> np.ndarray:
    """ Triangular function with unit area. """
    # Number of data points
    n = len(x)

    # Initialize probability values
    px = np.zeros(n)

    # Left side
    m = 1 / (xmode - xmin)
    b = 1 - m * xmode
    left_ndx = (x >= xmin) & (x < xmode)
    px[left_ndx] = m * x[left_ndx] + b

    # Peak
    peak_ndx = (x >= xmode) & (x <= xmode)
    px[peak_ndx] = 1.0

    # Right side
    m = -1 / (xmax - xmode)
    b = 0 - m * xmax
    right_ndx = (x > xmode) & (x <= xmax)
    px[right_ndx] = m * x[right_ndx] + b

    # Normalize area
    px /= sp.integrate.trapezoid(px, x)

    return px


#################### TRAPEZOIDAL FUNCTIONS ####################
def trapezoidal(
        x:np.ndarray, x1:float, x2:float, x3:float, x4:float) -> np.ndarray:
    """ Trapezoidal function with unit area. """
    # Number of data points
    n = len(x)

    # Initialize probability values
    px = np.zeros(n)

    # Left side
    m = 1 / (x2 - x1)
    b = 1 - m * x2
    left_ndx = (x > x1) * (x < x2)
    px[left_ndx] = m * x[left_ndx] + b

    # Boxcar
    boxcar_ndx = (x >= x2) * (x <= x3)
    px[boxcar_ndx] = 1.0

    # Right side
    m = -1 / (x4 - x3)
    b = 0 - m * x4
    right_ndx = (x > x3) & (x < x4)
    px[right_ndx] = m * x[right_ndx] + b

    # Normalize area
    px /= sp.integrate.trapezoid(px, x)

    return px


#################### GAUSSIAN FUNCTIONS ####################
def gaussian(x:np.ndarray, mu:float, sigma:float) -> np.ndarray:
    """ Gaussian function with unit area. """
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    f = np.exp(-0.5 * (x - mu)**2 / sigma**2)

    # Probability density
    px = a * f

    return px


def cumulative_gaussian(x:np.ndarray, mu:float, sigma:float) -> np.ndarray:
    """ Cumulative Gaussian function with unit area. """
    return 0.5 + 0.5 * sp.special.erf((x - mu) / (np.sqrt(2) * sigma))


# end of file