# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, 2026 Robert Zinke. Licensed under the MIT License.

# Public API
__all__ = [
    "PARAMETRIC_FUNCTIONS",
    "get_function_by_name",
    "get_cumulative_function_by_name",
    "check_number_inputs",
    "determine_min_max_limits",
]


# Import modules
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy as sp

from .. import precision
from . import value_arrays


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
    mass_below = xmin < x.min() - 10**-precision.RISER_PRECISION

    # Check for probability density above x domain
    mass_above = xmax > x.max() + 10**-precision.RISER_PRECISION

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
def _uniform_limits_(a, b):
    # Limits are defined by function min/max values
    return a, b


def uniform(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Uniform (boxcar) function with unit area.

    U(a, b) = f(x) = 1 / (b - a) for (a <= x <= b)
                     0 for (x < a or x > b)

    This assigns non-zeros values within the closed interval [a, b].

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    a : float
        Minimum value with non-zero probability density.
    b : float
        Maximum value with non-zero probability density.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Ensure a < b
    if not a < b:
        raise ValueError(
            f"`a` ({a}) must be strictly less than `b` ({b}) "
            f"for a uniform distribution"
        )

    # Domain checks
    xmin, xmax = _uniform_limits_(a, b)
    check_mass_against_value_range(x, xmin, xmax)

    # Initialize probability density values
    n = len(x)
    px = np.zeros(n)

    # Probability density values
    boxcar_ndx = (x >= a) & (x <= b)
    px[boxcar_ndx] = 1 / (b - a)

    return px


def cumulative_uniform(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Cumulative uniform function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    a : float
        Minimum value with non-zero probability density.
    b : float
        Maximum value with non-zero probability density.

    Returns
    -------
    Px : np.ndarray
        Cumulative probability values.
    """
    # Ensure a < b
    if not a < b:
        raise ValueError(
            f"`a` ({a}) must be strictly less than `b` ({b}) "
            f"for a uniform distribution"
        )

    # Number of data points
    n = len(x)

    # Initialize cumulative probability values
    Px = np.zeros(n)

    # Cumulative probability values
    boxcar_ndx = (x >= a) & (x <= b)
    Px[boxcar_ndx] = (x[boxcar_ndx] - a) / (b - a)
    Px[x > b] = 1.0

    return Px


def _triangular_limits_(a, c, b):
    # Limits are defined by left and right bases of triangle
    return a, b


def triangular(x: np.ndarray, a: float, c: float, b: float) -> np.ndarray:
    """Triangular function with unit area.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    a : float
        Left base of the triangle.
    c : float
        Peak of the triangle.
    b : float
        Right base of the triangle.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Ensure proper ordering
    if not a <= c <= b:
        raise ValueError(
            f"`a` ({a}) must be <= than `c` ({c}) must be <= `b` ({b})"
        )

    # Checks
    xmin, xmax = _triangular_limits_(a, c, b)
    check_mass_against_value_range(x, xmin, xmax)

    # Initialize probability density values
    n = len(x)
    px = np.zeros(n)

    # Left side
    left_ndx = (a <= x) & (x < c)
    px[left_ndx] = 2 * (x[left_ndx] - a) / ((b - a) * (c - a))

    # Peak
    peak_ndx = x == c
    px[peak_ndx] = 2 / (b - a)

    # Right side
    right_ndx = (c < x) & (x <= b)
    px[right_ndx] = 2 * (b - x[right_ndx]) / ((b - a) * (b - c))

    return px


def cumulative_triangular(
    x: np.ndarray, a: float, c: float, b: float
) -> np.ndarray:
    """Cumulative triangular function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    a : float
        Left base of the triangle.
    c : float
        Peak of the triangle.
    b : float
        Right base of the triangle.

    Returns
    -------
    px : np.ndarray
        Cumulative probability values.
    """
    # Ensure proper ordering
    if not a <= c <= b:
        raise ValueError(
            f"`a` ({a}) must be <= than `c` ({c}) must be <= `b` ({b})"
        )

    # Number of data points
    n = len(x)

    # Initialize cumulative probability values
    Px = np.zeros(n)

    # Left side
    left_ndx = (a < x) & (x <= c)
    Px[left_ndx] = (x[left_ndx] - a) ** 2 / ((b - a) * (c - a))

    # Right side
    right_ndx = (c < x) & (x < b)
    Px[right_ndx] = 1 - (b - x[right_ndx]) ** 2 / ((b - a) * (b - c))

    # Far right
    far_ndx = b <= x
    Px[far_ndx] = 1.0

    return Px


def _trapezoidal_limits_(a, b, c, d):
    # Limits are defined by left and right bases of trapezoid
    return a, d


def trapezoidal(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """Trapezoidal function with unit area.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    a : float
        Left base of trapezoid.
    b : float
        Left edge of boxcar portion.
    c : float
        Right edge of boxcar portion.
    d : float
        Right base of trapezoid.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Ensure proper ordering
    if not a <= b <= c <= d:
        raise ValueError(
            f"`a` ({a}) must be <= than `b` ({b}) "
            f"must be <= `c` ({c}) must be <= `d` ({d})"
        )

    # Checks
    xmin, xmax = _trapezoidal_limits_(a, b, c, d)
    check_mass_against_value_range(x, xmin, xmax)

    # Initialize probability density values
    n = len(x)
    px = np.zeros(n)

    # Normalization coefficient
    coef = 2 / (d + c - a - b)

    # Left side
    left_ndx = (a <= x) & (x < b)
    px[left_ndx] = coef * (x[left_ndx] - a) / (b - a)

    # Boxcar
    boxcar_ndx = (b <= x) & (x < c)
    px[boxcar_ndx] = coef

    # Right side
    right_ndx = (c <= x) & (x <= d)
    px[right_ndx] = coef * (d - x[right_ndx]) / (d - c)

    return px


def cumulative_trapezoidal(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """Cumulative trapezoidal function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    a : float
        Left base of trapezoid.
    b : float
        Left edge of boxcar portion.
    c : float
        Right edge of boxcar portion.
    d : float
        Right base of trapezoid.

    Returns
    -------
    px : np.ndarray
        Cumulative probability values.
    """
    # Ensure proper ordering
    if not a <= b <= c <= d:
        raise ValueError(
            f"`a` ({a}) must be <= than `b` ({b}) "
            f"must be <= `c` ({c}) must be <= `d` ({d})"
        )

    # Initialize cumulative probability values
    n = len(x)
    Px = np.zeros(n)

    # Common coefficient
    coef = 1 / (d + c - a - b)

    # Left side
    left_ndx = (a <= x) & (x < b)
    Px[left_ndx] = coef / (b - a) * (x[left_ndx] - a) ** 2

    # Boxcar
    boxcar_ndx = (b <= x) & (x < c)
    Px[boxcar_ndx] = coef * (2 * x[boxcar_ndx] - a - b)

    # Right side
    right_ndx = (c <= x) & (x <= d)
    Px[right_ndx] = 1 - coef / (d - c) * (d - x[right_ndx]) ** 2

    # Far right
    far_ndx = x > d
    Px[far_ndx] = 1.0

    return Px


def _gaussian_limits_(mu, sigma) -> tuple[float, float]:
    # Area of PDF to be covered
    target_coverage = sp.stats.norm.cdf(4)

    # Distance from mean at which the area is covered
    sigma_lim = sp.stats.norm.ppf(target_coverage)

    # Distances at which coverage is met
    xmin = mu - sigma * sigma_lim
    xmax = mu + sigma * sigma_lim

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
    check_mass_against_value_range(x, xmin, xmax)

    a = 1 / (sigma * np.sqrt(2 * np.pi))
    f = np.exp(-0.5 * (x - mu) ** 2 / sigma**2)

    # Probability density
    px = a * f

    return px


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


def _exponential_limits_(scale) -> tuple[float, float]:
    # Minimum distance
    xmin = 0

    # Area of PDF to be covered
    target_coverage = sp.stats.norm.cdf(4)

    # Distance from zero at which the area is covered
    xmax = scale * sp.stats.expon.ppf(target_coverage)

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
    nonnegative_ndx = x >= 0

    # Distribution components
    a = 1 / scale
    f = np.exp(-x[nonnegative_ndx] / scale)

    # Probability density
    px[nonnegative_ndx] = a * f

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
    positive_ndx = x > 0

    # Distribution components
    a = 1 / (x[positive_ndx] * sigma * np.sqrt(2 * np.pi))
    f = np.exp(-0.5 * (np.log(x[positive_ndx]) - mu) ** 2 / sigma**2)

    # Probability density
    px[positive_ndx] = a * f

    return px


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
    # Initialize cumulative probability values
    n = len(x)
    Px = np.zeros(n)

    # Indices over which function is non-zero
    positive_ndx = x > 0

    # Cumulative probability values
    Px[positive_ndx] = cumulative_gaussian(np.log(x[positive_ndx]), mu, sigma)

    return Px


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
    # Check degrees of freedom is positive
    if dof <= 0:
        raise ValueError(f"`dof` must be positive, got {dof}")

    # Checks
    xmin, xmax = _students_t_limits_(dof, mu, scale)
    check_mass_against_value_range(x, xmin, xmax)

    # Probability density
    px = sp.stats.t.pdf(x, df=dof, loc=mu, scale=scale)

    return px


def _delta_limits_(x0, dx) -> tuple[float, float]:
    xmin = x0 - dx
    xmax = x0 + dx

    # Target domain limits
    return xmin, xmax


def delta(x: np.ndarray, x0: float) -> np.ndarray:
    """Delta function with single spike at which all the probability resides.

    In theory, the delta function (or distribution) has infinite probability
    density at `x0`. However, this is not possible to define using array-based
    PDFs.
    Here, we define the probability density among as few values as possible
    (at most two) based on the spike position `x0` and the sample spacing `dx`.
    The defining property is that the CDF of the delta PDF is a step
    function from 0.0 to 1.0 at `x0`.

    The probability is split between the two samples bracketing `x0` in
    proportion to their proximity to `x0`, which preserves the mean of the
    function at exactly `x0` even if `x0` does not coincide with a sample.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    x0 : float
        Value at which event occurred.

    Returns
    -------
    px : np.ndarray
        Probability density values.
    """
    # Grid spacing from x-axis
    dx = value_arrays.sample_spacing_from_value_array(x)

    # Check spike lies at least one sample inside domain
    xmin, xmax = _delta_limits_(x0, dx)

    tolerance = 10**-precision.RISER_PRECISION
    if xmin < x.min() - tolerance or xmax > x.max() + tolerance:
        raise ValueError(
            f"x0 ({x0}) must lie at least one sample spacing ({dx}) inside "
            f"the domain ({x.min()}, {x.max()})"
        )

    # Initialize px array
    px = np.zeros(len(x))

    # Indices bracketing x0 to the left and right
    right_ndx = np.searchsorted(x, x0, side="left")
    left_ndx = right_ndx - 1

    # Domain values bracketing x0 to the left and right
    x_left = x[left_ndx]
    x_right = x[right_ndx]
    spacing = x_right - x_left

    # Probability density
    # Weights of the bracketing samples (sum to 1) divided by spacing so
    # PDF has unit area
    px[left_ndx] = (x_right - x0) / spacing**2
    px[right_ndx] = (x0 - x_left) / spacing**2

    return px


def cumulative_delta(x: np.ndarray, x0: float) -> np.ndarray:
    """Cumulative delta function.

    This is a step function analogous to a Heaviside function.

    Parameters
    ----------
    x : np.ndarray
        Value array over which to define the function.
    x0 : float
        Value at which event occurred.

    Returns
    -------
    Px : np.ndarray
        Cumulative probability values.
    """
    # Initialize cumulative probability values
    n = len(x)
    Px = np.zeros(n)

    # Step function for non-zero probability values
    Px[x >= x0] = 1.0

    return Px


@dataclass(frozen=True)
class ParametricFunction:
    """Class for defining a parametric function consistent with this library.

    Organize the necessary function definitions and properties.
    Return derived properties such as the number of function parameters.

    Attributes
    ----------
    name : str
        Name of the function (e.g., 'gaussian')
    function : Callable
        Parametric function definition.
    limits_function : Callable
        Method to find the suggested min/max limits of domain based on the
        function shape for a given set of parameters.
        Takes the distribution's parameters, plus `dx` as a keyword when
        `dx_required` is true.
    cumulative_function : Callable, optional
        Cumulative parametric function definition, if the cumulative function
        has a closed-form analytical definition.
    dx_required : bool, optional
        If true, the sample spacing `dx` must be explicitly specified.
    """

    name: str

    function: Callable

    limits_function: Callable

    cumulative_function: Callable | None = None

    dx_required: bool = False

    @property
    def reqd_args(self) -> list[str]:
        """
        Return a list of required parameters for the parametric function.
        """
        # Retrieve required arguments from function signature
        fcn_sig = inspect.signature(self.function)

        return [
            param.name
            for param in fcn_sig.parameters.values()
            if param.default is inspect.Parameter.empty
            and param.name != "x"
        ]

    def __call__(self, *args, **kwargs):
        """Call the parametric function itself."""
        return self.function(*args, **kwargs)


PARAMETRIC_FUNCTIONS: dict[str, ParametricFunction] = {
    "uniform": ParametricFunction(
        name="uniform",
        limits_function=_uniform_limits_,
        function=uniform,
        cumulative_function=cumulative_uniform,
    ),
    "triangular": ParametricFunction(
        name="triangular",
        limits_function=_triangular_limits_,
        function=triangular,
        cumulative_function=cumulative_triangular,
    ),
    "gaussian": ParametricFunction(
        name="gaussian",
        limits_function=_gaussian_limits_,
        function=gaussian,
        cumulative_function=cumulative_gaussian,
    ),
    "trapezoidal": ParametricFunction(
        name="trapezoidal",
        limits_function=_trapezoidal_limits_,
        function=trapezoidal,
        cumulative_function=cumulative_trapezoidal,
    ),
    "exponential": ParametricFunction(
        name="exponential",
        limits_function=_exponential_limits_,
        function=exponential,
    ),
    "lognormal": ParametricFunction(
        name="lognormal",
        limits_function=_lognormal_limits_,
        function=lognormal,
    ),
    "students_t": ParametricFunction(
        name="students_t",
        limits_function=_students_t_limits_,
        function=students_t,
    ),
    "delta": ParametricFunction(
        name="delta",
        limits_function=_delta_limits_,
        function=delta,
        cumulative_function=cumulative_delta,
        dx_required=True,
    ),
}


def _get_parametric_function_(function: str) -> ParametricFunction:
    """Return an object with the full parametric function definition.

    Raise if the function is not defined here.
    """
    # Check that the desired function is defined here
    if function not in PARAMETRIC_FUNCTIONS:
        raise ValueError(
            f"Function '{function}' is not defined. "
            f"Use one of {', '.join(PARAMETRIC_FUNCTIONS.keys())}"
        )

    return PARAMETRIC_FUNCTIONS[function]


def get_function_by_name(function: str) -> Callable[..., Any]:
    """Retrieve one of the parametric functions defined above by name.

    Parameters
    ----------
    function : str
        Parametric function name.

    Returns
    -------
    Callable
        Parameteric function.
    """
    # Return function
    return _get_parametric_function_(function).function


def get_cumulative_function_by_name(function: str) -> Callable[..., Any]:
    """Retrieve one of the cumulative parametric functions by name.

    Parameters
    ----------
    function : str
        Parametric function name.

    Returns
    -------
    Callable
        Cumulative parametric function.
    """
    # Retrieve parametric function
    para_fcn = _get_parametric_function_(function)

    # Check that closed-form analytical cumulative function is defined
    if para_fcn.cumulative_function is None:
        raise ValueError(
            f"Parametric function {function} is recognized, but does not have "
            f"a closed-form analytical expression defined here."
        )

    # Return function
    return para_fcn.cumulative_function


#################### CHECKS ####################
def check_number_inputs(function: str, variables: list[float]) -> None:
    """Check that the appropriate number of inputs are provided for the given
    function.

    Parameters
    ----------
    function : str
        Parametric function name.
    variables : list[float]
        Parameter values.

    Returns
    -------
    bool
        True if correct number of inputs provided.
    """
    # Retrieve required arguments from function signature
    n_vars_reqd = len(_get_parametric_function_(function).reqd_args)

    # Number of variables specified
    n_vars_specd = len(variables)

    # Check necessary number of values specified
    if n_vars_specd != n_vars_reqd:
        raise ValueError(
            f"{n_vars_reqd} values must be specified for a "
            f"{function} function, got {n_vars_specd}"
        )


def determine_min_max_limits(
    function: str,
    values: list[float],
    *,
    limit_positive: bool = False,
    dx: float | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Determine the minimum and maximum values of the PDF domain.

    Parameters
    ----------
    function : str
        Parametric function.
    values : list[float]
        Parameter values.
    limit_positive : bool, optional
        Limit the function limits to positive values only.
    dx : float, optional
        Sample spacing.

    Returns
    -------
    xmin : float
        Minimum value.
    xmax : float
        Maximum value.
    """
    # Retrieve parametric function
    para_fcn = _get_parametric_function_(function)

    # Determine min/max limits
    if para_fcn.dx_required:
        # Check that `dx` is specified if required
        if dx is None:
            raise ValueError(
                f"`dx` must be specified for a {function} function"
            )

        # Limits of function for which `dx` is required
        xmin, xmax = para_fcn.limits_function(*values, dx=dx)
    else:
        # Limits for generic function
        xmin, xmax = para_fcn.limits_function(*values)

    # Let minimum limit to positive
    if limit_positive:
        xmin = np.max([xmin, 0])

    # Report min/max value limits
    if verbose:
        print(
            f"Suggested minimum value {xmin:.4f}\n"
            f"Suggested maximum value {xmax:.4f}"
        )

    return xmin, xmax


# end of file
