# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import copy

import numpy as np

from riser.probability_functions import PDF
from riser import precision


#################### MOMENTS ####################
def expected_value(x:np.ndarray, px:np.ndarray, dx:float|np.ndarray) -> float:
    """Compute the expected value of a random variable X.
    Note the change in x can be explicitly defined as a single number
    (e.g., 0.01) or determined directly from the value array.

    Args    x - np.ndarray, values of X
            px - np.ndarray, relative probabilities of X
            dx - float or np.ndarray, change in x
    """
    return np.sum(x * px * dx)


def compute_raw_moment(
        x:np.ndarray, px:np.ndarray, dx:float|np.ndarray, n:int) -> float:
    """A raw moment is defined as:

    theta_n = integral(x^n * f(x) dx)

    This is typically only used to compute the mean, mu, for which n = 1,
    i.e., the expected value.

    Args    x - np.ndarray, values of X
            px - np.ndarray, relative probabilities of X
            dx - float or np.ndarray, change in x
            n - int, moment
    Returns theta_n - float, raw moment
    """
    theta_n = np.sum(x**n * px * dx)

    return theta_n


def compute_central_moment(
        x:np.ndarray, px:np.ndarray, dx:float|np.ndarray, n:int) -> float:
    """A central moment is computed about the function mean, mu:

    mu_n = integral((x - mu)^n * f(x) dx) = E[(X - mu)^n]

    Args    x - np.ndarray, values of X
            px - np.ndarray, relative probabilities of X
            dx - float or np.ndarray, change in x
            n - int, moment
    Returns mu_n - float, central moment
    """
    # Compute mean
    mu = expected_value(x, px, dx)

    # Compute central moment
    mu_n = expected_value((x - mu)**n, px, dx)

    return mu_n


def compute_standardized_moment(
        x:np.ndarray, px:np.ndarray, dx:float|np.ndarray, n:int) -> float:
    """A standardized moment is computed about the function mean, mu, and
    normalized by the standard deviation raised to the power of the moment.

    mu_std_n = mu_n / sigma^n = E[(X - mu)^n] / E[(X - mu)^2]^n/2

    Args    x - np.ndarray, values of X
            px - np.ndarray, relative probabilities of X
            dx - float or np.ndarray, change in x
            n - int, moment
    Returns mu_std_n - float, standardized moment
    """
    # Compute mean
    mu = expected_value(x, px, dx)

    # Compute central moment
    mu_std_n = expected_value((x - mu)**n, px, dx) \
            / expected_value((x - mu)**2, px, dx)**(n/2)

    return mu_std_n


#################### PDF STATISTICS ####################
def find_dx(pdf:PDF) -> np.ndarray:
    """Determine the change in x (dx) for a discrete PDF.
    In classical calculus, dx is a scalar number, which assumes that the
    function is regularly sampled.
    For the discrete PDFs used in practical applications, that might not be
    the case, i.e., the bin sizes may vary and a vector of bin sizes (dx's) is
    necessary.

    This routine returns a vector of bin sizes for all n-values in a PDF.
    The first n - 1 bin sizes are based on the differences from one x-value to
    the next. If the PDF is regularly sampled, the final bin size will be the
    same as the average bin size. If the PDF is irregularly sampled, the final
    bin size will be zero (excluding the final measurement).

    Args    pdf - PDF for which to determine dx
    Returns dx - np.ndarray
    """
    # Deteremine differences between x-samples
    diff_x = np.diff(pdf.x)

    # Determine regularity of sampling
    diff_x_std = np.std(diff_x)

    # Check regularity against machine error
    if diff_x_std > precision.RISER_PRECISION:
        # Irregular sampling of PDF
        return precision.fix_precision(np.diff(pdf.x, append=0))
    else:
        # Regular sampling
        return precision.fix_precision(np.diff(pdf.x, append=np.mean(diff_x)))


def pdf_mean(pdf:PDF) -> float:
    """Compute the mean of a PDF by integrating x * f(x).
    (Essentially a weighted average for the discrete PDF).

    This is the expected value E[X] of a random variable,
    and the first raw moment of a PDF.

    mu = E[X] = integral(x * f(x) * dx)

    Args    pdf - PDF to analyse
    Returns mean - float, mean of PDF
    """
    # Change in x
    dx = find_dx(pdf)

    # Compute expected value
    mu = expected_value(pdf.x, pdf.px, dx)

    return mu


def pdf_variance(pdf:PDF) -> float:
    """Compute the variance of a PDF.

    sigma2 = E[(X - mu)^2] = integral((x - mu)^2 * f(x) * dx)

    Args    pdf - PDF to analyse
    Returns variance - float, variance of PDF
    """
    # Change in x
    dx = find_dx(pdf)

    # Compute expected value
    mu = pdf_mean(pdf)

    # Compute variance
    sigma2 = np.sum((pdf.x - mu)**2 * pdf.px * dx)

    return sigma2


def pdf_std(pdf:PDF) -> float:
    """Compute the standard deviation (sigma) of a PDF.
    Standard deviation is the square root of the variance.

    sigma = sqrt(variance)

    Args    pdf - PDF to analyse
    Returns sigma - float, standard deviation of PDF
    """
    # Compute variance
    sigma2 = pdf_variance(pdf)

    # Compute standard deviation
    sigma = np.sqrt(sigma2)

    return sigma


def pdf_skewness(pdf:PDF) -> float:
    """Compute the skewness of a PDF.

    This is the standardized third central moment of a PDF.

    gamma = E[(X - mu)^3] / (E[(X - mu)^2])^3/2

    Args    pdf - PDF to analyse
    Returns gamma - float, skewness of PDF
    """
    # Change in x
    dx = find_dx(pdf)

    # Compute third standardized moment
    gamma = compute_standardized_moment(pdf.x, pdf.px, dx, n=3)

    return gamma


def pdf_kurtosis(pdf:PDF) -> float:
    """Compute the kurtosis, or "tailedness"/"peakiness", of a PDF.

    This is the standardized fourth central moment of a PDF.

    kappa = E[(X - mu)^4] / (E[(X - mu)^2])^4/2

    Args    pdf - PDF to analyse
    Returns kappa - float, kurtosis of PDF
    """
    # Change in x
    dx = find_dx(pdf)

    # Compute third standardized moment
    kappa = compute_standardized_moment(pdf.x, pdf.px, dx, n=4)

    return kappa


def pdf_mode(pdf:PDF) -> float:
    """Determine the mode (peak value) of a PDF.

    Args    pdf - PDF to analyse
    Returns mode - float, mode of PDF
    """
    return np.mean(pdf.x[pdf.px == pdf.px.max()])


def pdf_median(pdf:PDF) -> float:
    """Compute the median of a PDF based on the value where the CDF is 0.5.

    Args    pdf - PDF to analyse
    Returns median - float, median of PDF
    """
    return pdf.inversse_transform(0.5).item()


#################### STATISTICAL SUMMARIES ####################
class PDFstatistics:
    def __init__(self, pdf:PDF):
        """Compute basic statistical properties of a PDF.
        """
        # Compute location statistics
        self.mode = pdf_mode(pdf)
        self.median = pdf_median(pdf)

        # Compute moments
        self.mean = pdf_mean(pdf)
        self.std = pdf_std(pdf)
        self.variance = pdf_variance(pdf)
        self.skewness = pdf_skewness(pdf)
        self.kurtosis = pdf_kurtosis(pdf)

    def __str__(self):
        print_str = (f"  mode: {self.mode:.3f}"
                     f"\nmedian: {self.median:.3f}"
                     f"\n  mean: {self.mode:.3f}"
                     f"\n   std: {self.std:.3f}"
                     f"\n   var: {self.variance:.3f}"
                     f"\n  skew: {self.skewness:.3f}"
                     f"\n  kurt: {self.kurtosis:.3f}")

        return print_str


#################### CONFIDENCE RANGES ####################
class ConfidenceInterval:
    """Convenience class to store and report confidence intervals of a PDF.
    """
    def __init__(self, pdf_name:str, confidence:float,
                 values:tuple|list[tuple], method:str=None):
        # Record arguments
        self.pdf_name = pdf_name
        self.method = method

        self.confidence = confidence
        self.values = values

    def __str__(self):
        print_str = f"{self.pdf_name}"

        print_str += f"\n{self.confidence * 100} % confidence"
        if self.method is not None:
            print_str += f" ({method})"

        return print_str


def compute_interquantile_range(pdf:PDF,
        confidence:float=0.6828) -> (float, float):
    """Compute the interquantile range (IQR) values of a PDF based on the CDF.

    Args    pdf - PDF to analyse
            confidence - float, confidence level
    Returns
    """
    # Determine the lower and upper confidence levels
    half_confidence = confidence / 2
    lower = 0.5 - half_confidence
    upper = 0.5 + half_confidence

    # Compute the CDF value for each confidence level
    values = (pdf.inversse_transform(lower), pdf.inversse_transform(upper))

    return values


def compute_highest_posterior_density(pdf:PDF,
        confidence:float=0.6828) -> list[tuple[float]]:
    """Compute the highest posterior density (HPD) values of a PDF.

    Args    pdf - PDF to analyse
            confidence - list[float], confidence levels
    Returns
    """
    # Value index numbers
    val_nbs = np.array([*range(len(pdf))])

    # Compute probabilities
    dx = find_dx(pdf)
    p_i = pdf.px * dx

    # Sort the probabilities from largest to smallest
    sort_ndx = np.argsort(p_i)
    sort_ndx = sort_ndx[::-1]

    vals_sort = val_nbs[sort_ndx]
    x_sort = pdf.x[sort_ndx]
    px_sort = pdf.px[sort_ndx]
    p_i_sort = p_i[sort_ndx]

    # Sum probabilities until they reach the specified confidence limit
    P_sort = np.cumsum(p_i_sort)

    # Determine which values meet confidence bounds
    conf_ndxs = (P_sort <= confidence)

    # Keep x, px value probability pairs that are within confidence limits
    x_sort_conf = x_sort[conf_ndxs]
    px_sort_conf = px_sort[conf_ndxs]
    vals_sort_conf = vals_sort[conf_ndxs]

    # Un-sort values in confidence limit by x-value
    unsort_ndx = np.argsort(x_sort_conf)

    x_conf = x_sort_conf[unsort_ndx]
    px_conf = px_sort_conf[unsort_ndx]
    vals_conf = vals_sort_conf[unsort_ndx]

    # Number of values in confidence limit
    n_conf = len(x_conf)

    # Group values by continuity
    clusters = []
    cluster_start = x_conf[0]
    for i in range(1, n_conf):
        if any([(vals_conf[i] - vals_conf[i-1]) > 1,
                (i == n_conf-1)]):
            # Cluster end
            cluster_end = x_conf[i-1]

            # Record cluster
            clusters.append((cluster_start, cluster_end))

            # Start next cluster
            cluster_start = x_conf[i]

    return clusters


# end of file