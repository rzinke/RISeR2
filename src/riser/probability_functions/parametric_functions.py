# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
PARAMETRIC_FUNCTION_PARAM_NBS = {
    "boxcar": 2,
    "triangular": 3,
    "trapezoidal": 4,
    "gaussian": 2
}

PARAMETRIC_FUNCTIONS = tuple(PARAMETRIC_FUNCTION_PARAM_NBS.keys())


# Import modules
import numpy as np
import scipy as sp


#################### PARAMETRIC FUNCTIONS ####################
def boxcar(x:np.ndarray, xmin:float, xmax:float) -> np.ndarray:
    """Boxcar function with unit area.
    """
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


def triangular(
        x:np.ndarray, xmin:float, xmode:float, xmax:float
    ) -> np.ndarray:
    """Triangular function with unit area.
    """
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


def trapezoidal(
        x:np.ndarray, x1:float, x2:float, x3:float, x4:float
    ) -> np.ndarray:
    """Trapezoidal function with unit area.
    """
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


def gaussian(x:np.ndarray, mu:float, sigma:float) -> np.ndarray:
    """Gaussian function with unit area.
    """
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    f = np.exp(-0.5 * (x - mu)**2 / sigma**2)

    # Probability density
    px = a * f

    return px


def cumulative_gaussian(x:np.ndarray, mu:float, sigma:float) -> np.ndarray:
    """Cumulative Gaussian function with unit area.
    """
    return 0.5 + 0.5 * sp.special.erf((x - mu) / (np.sqrt(2) * sigma))


#################### FUNCTION RETRIEVAL ####################
def get_function_by_name(fcn_name:str):
    """Retrieve one of the parametric functions defined above by name.

    Args    fcn_name - str, function name
    Returns fcn - parameteric function
    """
    # Check that the desired function is defined here
    if fcn_name not in PARAMETRIC_FUNCTIONS:
        raise ValueError(f"Function {fcn_name} is not defined")

    # Return function
    if fcn_name == "boxcar":
        return boxcar

    elif fcn_name == "triangular":
        return triangular

    elif fcn_name == "trapezoidal":
        return trapezoidal

    elif fcn_name == "gaussian":
        return gaussian

    else:
        return None


#################### CHECKS ####################
def check_number_inputs(distribution:str, values:list[float]) -> bool:
    """Check that the appropriate number of inputs are provided for the given
    distribution.

    Args    distribution - str, parametric function
            values - list[float], parameter values
    Returns bool, True if correct number of inputs provided
    """
    # Number of values required
    n_vals_reqd = PARAMETRIC_FUNCTION_PARAM_NBS[distribution]

    # Number of values specified
    n_vals_specd = len(values)

    # Check necessary number of values specified
    if n_vals_specd != n_vals_reqd:
        raise Exception(
            f"{n_vals_reqd} must be specified for a {distribution} distribution"
        )

    return True


def determine_min_max_limits(distribution:str, values:list[float],
        limit_positive:bool=False, verbose=False) -> (float, float):
    """Determine the minimum and maximum values of the PDF domain.

    Args    distribution - str, parametric function
            values - list[float], parameter values
    Returns xmin, xmax - float, min/max values
    """
    # Behave based on function type
    if distribution in ["boxcar", "triangular", "trapezoidal"]:
        # Use first and last values
        xmin = values[0]
        xmax = values[-1]

    elif distribution in ["gaussian"]:
        # Parse values
        mu, sigma = values

        # Use 4-sigma limit
        sigma_lim = 4 * sigma

        # Limit at zero
        xmin = (
            np.max([mu - sigma_lim, 0]) if limit_positive == True 
            else mu - sigma_lim
        )

        # Max value
        xmax = mu + sigma_lim

    # Report if requested
    if verbose == True:
        print(f"Minimum value {xmin}\nMaximum value {xmax}")

    return xmin, xmax


# end of file