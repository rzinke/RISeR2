# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Constants
PDF_FORMATION_METHODS = (
    "histogram",
    "kde",
)

# Import modules
import numpy as np
import scipy as sp

import riser.probability_functions as PDFs


#################### FORMATION METHODS ####################
def samples_to_pdf_histogram(
    samples:np.ndarray,
    *,
    xmin: float | None=None,
    xmax: float | None=None,
    dx: float | None=None,
    name: str | None=None,
    variable_type: str | None=None,
    unit: str | None=None,
    verbose=False
) -> PDFs.PDF:
    """Form discrete samples into a PDF by binning them into a histogram.
    Note: The number of histogram values will be 1 less than the number of bin
    edges, leaving the question of what value is represented by each probability
    density.
    Here, the probability densities are set to correspond to the left edge of
    each bin and the final bin is set to zero. This is because, for slip rate
    estimates, the smaller values should be preserved and the larger values
    trail toward zero. The bins are defined as half-open, [), where the left
    value is included.

    Args    samples - np.ndarray, discrete samples
            xmin, xmax - float, min/max values
            dx - float, value-step
            name - str, brief descriptive identifier
            variable_type - str, sampled quantity, e.g., age, displacement
            unit - str, value unit
    Returns pdf - Empirical PDF based on samples
    """
    if verbose == True:
        print("Converting samples to PDF using histogram method")

    # Determine value limits
    xmin = np.min(samples) if xmin is None else xmin
    xmax = np.max(samples) if xmax is None else xmax

    # Determine bin sizes
    n_samples = len(samples)
    dx = np.sqrt(n_samples) if dx is None else dx

    # Create histogram value array
    x = PDFs.value_arrays.create_precise_value_array(xmin, xmax, dx)

    # Bin points in histogram
    px, _ = np.histogram(samples, bins=x, density=True)

    # Handle edge cases
    px = np.pad(px, (0, 1), 'constant')

    # Form histogram data into PDF
    pdf = PDFs.PDF(
        x=x,
        px=px,
        name=name,
        variable_type=variable_type,
        unit=unit
    )

    return pdf


def samples_to_pdf_kde(
    samples:np.ndarray,
    *,
    xmin: float|None =None,
    xmax: float=None,
    dx: float|None =None,
    name: str|None =None,
    variable_type: str|None =None,
    unit: str|None =None,
    verbose=False
) -> PDFs.PDF:
    """Form discrete samples into a PDF using kernel density estimation (KDE)
    with a Gaussian kernel.
    This method is not recommended for converting slip rate samples into PDFs
    because slip rates skew toward faster values, and an adaptive kernel
    is necessary to adjust the bandwidth between more frequent samples at
    smaller values, to sparser samples at larger values.

    Args    samples - np.ndarray, discrete samples
            xmin, xmax - float, min/max values
            dx - float, value-step
            name - str, brief descriptive identifier
            variable_type - str, sampled quantity, e.g., age, displacement
            unit - str, value unit
    Returns pdf - Empirical PDF based on samples
    """
    if verbose == True:
        print("Converting samples to PDF using KDE")

    # Determine value limits
    xmin = np.min(samples) if xmin is None else xmin
    xmax = np.max(samples) if xmax is None else xmax

    # Determine bin sizes
    n_samples = len(samples)
    dx = np.sqrt(n_samples) if dx is None else dx

    # Create histogram value array
    x = value_arrays.create_precise_value_array(xmin, xmax, dx)

    # Compute KDE
    kde = sp.stats.gaussian_kde(samples)

    # Interpolate along value array
    px = kde.pdf(x)

    # Form histogram data into PDF
    pdf = PDF(
        x=x,
        px=px,
        name=name,
        variable_type=variable_type,
        unit=unit
    )

    return pdf


def get_pdf_formation_function(method: str, verbose=False):
    """Retrieve a PDF formation function by name.
    """
    # Format method input
    method = method.lower()

    # Check that method is valid
    if method not in PDF_FORMATION_METHODS:
        raise ValueError(
            f"Method {method} not supported. Must be one of "
            f"{', '.join(PDF_FORMATION_METHODS)}"
        )

    # Report if requested
    if verbose == True:
        print(f"PDF formation method: {method}")

    # Return method
    if method == "histogram":
        return samples_to_pdf_histogram

    elif method == "kde":
        return samples_to_pdf_kde


# end of file