# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved


# Import modules
import numpy as np

from riser.probability_functions import ProbabilityDensityFunctions as PDFs


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
def pdf_dx(pdf:PDFs.ProbabilityDensityFunction) -> np.ndarray:
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
    if diff_x_std > 1E-12:
        # Irregular sampling of PDF
        return np.diff(pdf.x, append=0)
    else:
        # Regular sampling
        return np.diff(pdf.x, append=np.mean(diff_x))


def pdf_mean(pdf:PDFs.ProbabilityDensityFunction) -> float:
    """Compute the mean of a PDF by integrating x * f(x).
    (Essentially a weighted average for the discrete PDF).

    This is the expected value E[X] of a random variable,
    and the first raw moment of a PDF.

    mu = E[X] = integral(x * f(x) * dx)

    Args    pdf - PDF to analyse
    Returns mean - float, mean of PDF
    """
    # Change in x
    dx = pdf_dx(pdf)

    # Compute expected value
    mu = expected_value(pdf.x, pdf.px, dx)

    return mu


def pdf_variance(pdf:PDFs.ProbabilityDensityFunction) -> float:
    """Compute the variance of a PDF.

    sigma2 = E[(X - mu)^2] = integral((x - mu)^2 * f(x) * dx)

    Args    pdf - PDF to analyse
    Returns variance - float, variance of PDF
    """
    # Change in x
    dx = pdf_dx(pdf)

    # Compute expected value
    mu = pdf_mean(pdf)

    # Compute variance
    sigma2 = np.sum((pdf.x - mu)**2 * pdf.px * dx)

    return sigma2


def pdf_std(pdf:PDFs.ProbabilityDensityFunction) -> float:
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


def pdf_skewness(pdf:PDFs.ProbabilityDensityFunction) -> float:
    """Compute the skewness of a PDF.

    This is the standardized third central moment of a PDF.

    gamma = E[(X - mu)^3] / (E[(X - mu)^2])^3/2

    Args    pdf - PDF to analyse
    Returns gamma - float, skewness of PDF
    """
    # Change in x
    dx = pdf_dx(pdf)

    # Compute third standardized moment
    gamma = compute_standardized_moment(pdf.x, pdf.px, dx, n=3)

    return gamma


def pdf_kurtosis(pdf:PDFs.ProbabilityDensityFunction) -> float:
    """Compute the kurtosis, or "tailedness"/"peakiness", of a PDF.

    This is the standardized fourth central moment of a PDF.

    kappa = E[(X - mu)^4] / (E[(X - mu)^2])^4/2

    Args    pdf - PDF to analyse
    Returns kappa - float, kurtosis of PDF
    """
    # Change in x
    dx = pdf_dx(pdf)

    # Compute third standardized moment
    kappa = compute_standardized_moment(pdf.x, pdf.px, dx, n=4)

    return kappa


def pdf_mode(pdf:PDFs.ProbabilityDensityFunction) -> float:
    """Determine the mode (peak value) of a PDF.

    Args    pdf - PDF to analyse
    Returns mode - float, mode of PDF
    """
    return pdf.x[pdf.px == pdf.px.max()]


def pdf_median(pdf:PDFs.ProbabilityDensityFunction) -> float:
    """Compute the median of a PDF based on the value where the CDF is 0.5.

    Args    pdf - PDF to analyse
    Returns median - float, median of PDF
    """
    return pdf.compute_cdf_value(0.5)



#################### CONFIDENCE RANGES ####################
class ConfidenceValues:
    """Class to conveniently hold confidence statistics.
    """
    def __init__(self, confidences:list[(float, float)],
                 method:str, pdf_name:str=None):
        """
        """
        # Record confidence level-value pairs
        self.confidences = confidences

        # Record descriptive parameters
        self.pdf_name = pdf_name
        self.method = method

    def __str__(self):
        print_str = "Confidence values:"

        if self.pdf_name is not None:
            print_str += f"\nPDF: {self.pdf_name}"

        if self.method is not None:
            print_str += f" ({self.method})"

        for level, value in self.confidences:
            print_str += f"\n\t{level:.3f}: {value:.3f}"

        return print_str


def compute_interquantile_range(pdf:PDFs.ProbabilityDensityFunction,
                                confidence_levels:list[float],
                                pdf_name:str=None,
                                verbose=False) -> "ConfidenceValues":
    """Compute the interquantile range (IQR) values of a PDF based on the CDF.

    Args    pdf - PDF to analyse
            confidence - list[float], confidence levels
    Returns confidences - ConfidenceValues
    """
    # Compute the CDF value for each confidence level
    values = [pdf.pit(conf) for conf in confidence_levels]

    # Format in ConfidenceValues object
    confidences = ConfidenceValues(zip(confidence_levels, values),
                                   method="IQR",
                                   pdf_name=pdf_name)

    # Report if requested
    if verbose == True:
        print(confidences)

    return confidences


#################### STATISTICAL SUMMARIES ####################
class PDFstatistics:
    """
    """
    pass


# end of file