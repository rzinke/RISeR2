# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "PARAMETRIC_FUNCTIONS",
    "boxcar",
    "triangular",
    "gaussian",
    "exponential",
    "lognormal",
    "get_function_by_name",
    "check_number_inputs",
    "determine_min_max_limits",
    "cumulative_gaussian",
    "cumulative_lognormal",
]


# Import modules
import warnings
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy as sp

from .. import (
    precision,
    integration,
)


#################### SUPPORT FUNCTIONS ####################
def check_mass_against_value_range(
    x: np.ndarray, xmin: float, xmax: float
) -> None:
    """Check that the xmin and xmax values of the PDF lie within the value
    range of x.

    Parameters
    ----------
    x : np.ndarray
        Value array over which the function is defined.
    xmin : float
        Theoretical minimum value of significance to check against value array.
    xmax : float
        Theoretical maximum value of significance to check against value array.
    """
    # Check for probability density below x domain
    mass_below = (xmin < x.min() - 10**-precision.RISER_PRECISION)

    # Check for probability density above x domain
    mass_above = (xmax > x.max() + 10**-precision.RISER_PRECISION)

    if mass_below and not mass_above:
        warnings.warn(
            "Significant probability lies below the specified PDF value range"
        )
    elif mass_above and not mass_below:
        warnings.warn(
            "Significant probability lies above the specified PDF value range"
        )
    elif mass_above and mass_below:
        warnings.warn(
            "Significant probability lies outside the specified PDF value range"
        )


#################### PARAMETRIC FUNCTIONS ####################
def boxcar(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Boxcar function with unit area.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    xmin : float
        Minimum value with non-zero probability density.
    xmax : float
        Maximum value with non-zero probability density.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Checks
    check_mass_against_value_range(x, xmin, xmax)

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
    x:np.ndarray, xmin: float, xmode: float, xmax: float
) -> np.ndarray:
    """Triangular function with unit area.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    xmin : float
        Minimum value with non-zero probability density.
    xmode : float
        Value of peak probability density.
    xmax : float
        Maximum value with non-zero probability density.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Ensure proper ordering
    if not xmin <= xmode <= xmax:
        raise ValueError(
            f"`xmin` ({xmin}) must be <= than `xmode` ({xmode}) "
            f"must be <= `xmax` ({xmax})"
        )

    # Checks
    check_mass_against_value_range(x, xmin, xmax)

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

    # Ensure all values > 0 (rounding error)
    px[px < 0] = 0

    # Normalize based on area of triangle
    px *= 2 / (xmax - xmin)

    return px


def trapezoidal(
    x: np.ndarray, x1: float, x2: float, x3: float, x4: float
) -> np.ndarray:
    """Trapezoidal function with unit area.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    x1 : float
        Minimum value with non-zero probability density.
    x2 : float
        Minimum value of the boxcar portion of the function.
    x3 : float
        Maximum value of the boxcar portion of the function.
    x4 : float
        Maximum value with non-zero probability density.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Ensure proper ordering
    if not x1 <= x2 <= x3 <= x4:
        raise ValueError(
            f"`x1` ({x1}) must be <= than `x2` ({x2}) "
            f"must be <= `x3` ({x3}) must be <= `x4` ({x4})"
        )

    # Checks
    check_mass_against_value_range(x, x1, x4)

    # Initialize probability density values
    n = len(x)
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
    px /= integration.integrate(x=x, px=px)

    return px


def _gaussian_limits_(mu, sigma) -> tuple[float, float]:
    # Area of PDF to be covered
    target_coverage = sp.stats.norm.cdf(4)

    # Distance from mean at which the area is covered
    sigma_lim = sp.stats.norm.ppf(target_coverage)

    # Distances at which coverage is met
    xmin = mu - sigma_lim
    xmax = mu + sigma_lim

    # Target domain limits
    return xmin, xmax

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    mu : float
        Center of the Gaussian function.
    sigma : float
        Breadth (standard deviation) of the Gaussian function.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Checks
    xmin, xmax = _gaussian_limits_(mu, sigma)
    check_mass_against_value_range(x, mu - 4 * sigma, mu + 4 * sigma)

    a = 1 / (sigma * np.sqrt(2 * np.pi))
    f = np.exp(-0.5 * (x - mu)**2 / sigma**2)

    # Probability density
    px = a * f

    return px


def _exponential_limits_(scale) -> tuple[float, float]:
    # Minimum distance
    xmin = 0

    # Area of PDF to be covered
    target_coverage = sp.stats.norm.cdf(4)

    # Distance from zero at which the area is covered
    xmax = sp.stats.expon.ppf(target_coverage)

    return xmin, xmax

def exponential(x: np.ndarray, scale: float) -> np.ndarray:
    """Exponential function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    scale : float
        Scale parameter of the exponential function.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Checks
    xmin, xmax = _exponential_limits_(scale)
    check_mass_against_value_range(x, xmin, xmax)

    # Initialize probability density values
    n = len(x)
    px = np.zeros(n)

    # Indices over which function is non-zero
    nonzero_ndx = x > 0

    # Distribution components
    a = 1 / scale
    f = np.exp(-x[nonzero_ndx] / scale)

    # Probability density
    px[nonzero_ndx] = a * f

    return px


def _lognormal_limits_(mu, sigma) -> tuple[float, float]:
    # Minimum distance
    xmin = 0

    # Area of PDF to be covered
    target_coverage = sp.stats.norm.cdf(4)

    # Distance from zero at which the area is covered
    xmax = np.exp(mu + sp.stats.norm.ppf(target_coverage) * sigma)

    return xmin, xmax

def lognormal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Log-normal function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    mu : float
        Center of the Gaussian function.
    sigma : float
        Breadth (standard deviation) of the Gaussian function.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Checks
    xmin, xmax = _lognormal_limits_(mu, sigma)
    check_mass_against_value_range(x, xmin, xmax)

    # Initialize probability density values
    n = len(x)
    px = np.zeros(n)

    # Indices over which function is non-zero
    nonzero_ndx = x > 0

    # Distribution components
    a = 1 / (x[nonzero_ndx] * sigma * np.sqrt(2 * np.pi))
    f = np.exp(-0.5 * (np.log(x[nonzero_ndx]) - mu)**2 / sigma**2)

    # Probability density
    px[nonzero_ndx] = a * f

    return px


def _students_t_limits_(dof, mu, scale) -> tuple[float, float]:
    # Target coverage - 0.99997
    target_coverage = sp.stats.norm.cdf(4)

    # Distances from mean at which the area is covered
    sigma_lim = sp.stats.t.ppf(target_coverage, df=dof) * scale

    # Distances at which coverage is met
    xmin = mu - sigma_lim
    xmax = mu + sigma_lim

    # Target domain limits
    return xmin, xmax

def students_t(
    x: np.ndarray, dof: float, mu: float, scale: float
) -> np.ndarray:
    """Student's t-distribution function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    dof : float
        Degrees of freedom (N - 1 for sample size N)
    mu : float
        Location of the distribution (sample mean).
    scale : float
        Scale parameter of the exponential function
        (standard error of the mean).

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Checks
    xmin, xmax = _students_t_limits_(dof, mu, scale)
    check_mass_against_value_range(x, xmin, xmax)

    # Probability density
    px = sp.stats.t.pdf(x, df=dof, loc=mu, scale=scale)

    return px


PARAMETRIC_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "boxcar": boxcar,
    "triangular": triangular,
    "trapezoidal": trapezoidal,
    "gaussian": gaussian,
    "exponential": exponential,
    "lognormal": lognormal,
    "students_t": students_t,
}


def get_function_by_name(distribution: str) -> Callable[..., Any]:
    """Retrieve one of the parametric functions defined above by name.

    Parameters
    ----------
    distribution : str
        Parametric function name.

    Returns
    -------
    fcn : Callable
        Parameteric function.
    """
    # Check that the desired function is defined here
    if distribution not in PARAMETRIC_FUNCTIONS:
        raise ValueError(
            f"Function '{distribution}' is not defined. "
            f"Use one of {', '.join(PARAMETRIC_FUNCTIONS.keys())}"
        )

    # Return function
    return PARAMETRIC_FUNCTIONS[distribution]


#################### CHECKS ####################
def check_number_inputs(distribution: str, variables: list[float]) -> None:
    """Check that the appropriate number of inputs are provided for the given
    distribution.

    Parameters
    ----------
    distribution : str
        Parametric function name.
    variables : list[float]
        Parameter values.

    Returns
    -------
    bool
        True if correct number of inputs provided.
    """
    # Retrieve required arguments from function signature
    fcn_sig = inspect.signature(PARAMETRIC_FUNCTIONS[distribution])
    reqd_args = [
        param.name for param in fcn_sig.parameters.values()
        if param.default is inspect.Parameter.empty
    ]

    # Number of values required
    n_vals_reqd = len(reqd_args)

    # Number of variables required is one less than the number of arguments
    # because x is not counted here
    n_vars_reqd = n_vals_reqd - 1

    # Number of variables specified
    n_vars_specd = len(variables)

    # Check necessary number of values specified
    if n_vars_specd != n_vars_reqd:
        raise ValueError(
            f"{n_vars_reqd} values must be specified for a "
            f"{distribution} distribution, got {n_vars_specd}"
        )


def determine_min_max_limits(
    distribution: str,
    values: list[float],
    *,
    limit_positive: bool = False,
    verbose: bool = False,
) -> tuple[float, float]:
    """Determine the minimum and maximum values of the PDF domain.

    Parameters
    ----------
    distribution : str
        Parametric function.
    values : list[float]
        Parameter values.
    limit_positive : bool, optional
        Limit the function limits to positive values only.

    Returns
    -------
    xmin : float
        Minimum value.
    xmax : float
        Maximum value.
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

        # Min value
        xmin = mu - sigma_lim

        # Limit at zero
        if limit_positive:
            xmin = np.max([xmin, 0])

        # Max value
        xmax = mu + sigma_lim

    elif distribution in ["exponential"]:
        # Parse values
        scale = values[0]

        # Min value
        xmin = 0

        # Max value
        xmax = 10 * scale

    elif distribution in ["lognormal"]:
        # Parse values
        mu, sigma = values

        # Min value
        xmin = 0

        # Max value
        xmax = np.exp(mu + 4 * sigma)

    else:
        raise ValueError(
            f"Min/max limits for distribution '{distribution}' not supported"
        )

    # Report if requested
    if verbose:
        print(f"Minimum value {xmin}\nMaximum value {xmax}")

    return xmin, xmax


#################### PARAMETRIC CDFS ####################
def cumulative_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Cumulative Gaussian function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    mu : float
        Center of the Gaussian function.
    sigma : float
        Breadth (standard deviation) of the Gaussian function.

    Returns
    -------
    Px : np.ndarray
        Cumulative probability values.
    """
    return 0.5 + 0.5 * sp.special.erf((x - mu) / (np.sqrt(2) * sigma))


def cumulative_lognormal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Cumulative log-normal function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    mu : float
        Center of the Gaussian function.
    sigma : float
        Breadth (standard deviation) of the Gaussian function.

    Returns
    -------
    Px : np.ndarray
        Cumulative probability values.
    """
    return cumulative_gaussian(np.log(x), mu, sigma)


# end of file
