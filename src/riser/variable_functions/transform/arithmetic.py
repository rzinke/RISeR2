# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
These functions carry out arithmetic between variables:
  - addition, i.e., X1 + X2
  - subtraction, i.e., X1 - X2
  - multiplication (product distribution), i.e., X1 * X2
  - division (ratio distribution), i.e., X1 / X2

and a function to negate (i.e., -1 * X1) a variable.

This module also includes bespoke convolution functions that explicitly show
the mechanics of how convolution is implemented.
These functions are provided only for reference because they are many times
slower than np.convolve.
"""


# Public API
__all__ = [
    "negate_variable",
    "add_variables",
    "subtract_variables",
    "multiply_variables",
    "divide_variables",
]


# Import modules
import warnings

import numpy as np

from ... import (
    precision,
    probability_functions as PDFs,
)


#################### GENERIC FUNCTIONS ####################
def convolve_input_side(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolution operator formulated from the input side.

    Parameters
    ----------
    x : np.ndarray
        Array to convolve with h.
    h : np.ndarray
        Array to convolve with x.
    
    Returns
    -------
    y : np.ndarray
        Convolved array.
    """
    # Array lengths
    nx = len(x)
    nh = len(h)
    ny = nx + nh - 1

    # Pre-allocate output array
    y = np.zeros(ny)

    # Loop through first array
    for i in range(nx):
        for j in range(nh):
            y[i + j] += x[i] * h[j]

    return y


def convolve_output_side(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolution operator formulated from the output side.

    Parameters
    ----------
    x : np.ndarray
        Array to convolve with h.
    h : np.ndarray
        Array to convolve with x.
    
    Returns
    -------
    y : np.ndarray
        Convolved array.
    """
    # Array lengths
    nx = len(x)
    nh = len(h)
    ny = nx + nh - 1

    # Pre-allocate output array
    y = np.zeros(ny)

    # Loop through output array
    for i in range(ny):
        # Loop through filter array
        for j in range(nh):
            # Check if valid
            if (i - j >= 0) and (i - j < nx):
                y[i] += x[i - j] * h[j]

    return y


#################### RANDOM VARIABLE ARITHMETIC ####################
def negate_variable(
    pdf: PDFs.PDF,
    verbose: bool = False,
) -> PDFs.PDF:
    """Negate a random variable expressed as a PDF.

    Negating the x-values, and flip the probability densities left for right.

    Parameters
    ----------
    pdf : PDF
        PDF to negate.
    
    Returns
    -------
    neg_pdf : PDF
        Negated PDF.
    """
    if verbose:
        print("Negate PDF")

    # Negate values
    neg_x = -pdf.x[::-1]

    # Flip probability densities
    neg_px = pdf.px[::-1]

    # Formulate output name
    neg_name = f"(negative) {pdf.name}" if pdf.name is not None else None

    # Compose metadata
    metadata_dict = pdf.metadata.as_dict()
    metadata_dict["name"] = neg_name

    # Form results into PDF
    neg_pdf = PDFs.PDF(
        x=neg_x,
        px=neg_px,
        **metadata_dict,
    )

    return neg_pdf


def add_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    *,
    name: str | None = None,
    verbose: bool = False,
) -> PDFs.PDF:
    """Add random variables PDF1 (X) and PDF2 (Y) to get a PDF of the sum of
    their values (Z).

    Theory:
    For discrete PDFs, variable addition is a sum of joint probabilties as a
    function of values. This is exactly convolution, and is mathematically best
    expressed from the "output side".

        P(Z = z) = sum(P(X = k).P(Y = z - k))
        or
        fZ(z) = integral(fX(x).fY(z - x) dx)

    Machinery:
    This function takes two PDFs that will be sampled on the same value axis.
    It creates an output array based on the input PDFs values, with the
    minimum sum being twice the minimum input, and the maximum sum being twice
    the maximum input.

    It then computes the probability density at each summed value using
    NumPy's own convolution (np.convolve, mode="full") for speed. See
    convolve_output_side for an explicit, unoptimized implementation of the
    identical output-side convolution algorithm described above, useful for
    understanding or reimplementing the mechanics directly.

    Parameters
    ----------
    pdf1 : PDF
        PDF to add to pdf2.
    pdf2 : PDF
        PDF to add to pdf1.
    name : str, optional
        Name of summed PDF.
    
    Returns
    -------
    pdf_sum : PDF
        Summed PDF.
    """
    if verbose:
        print("Adding variables")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata(
        [pdf1.metadata, pdf2.metadata], name=name,
    )

    # Parameters
    x_min = pdf1.x[0]
    x_max = pdf1.x[-1]
    nx = len(pdf1)

    # Output array length
    nz = 2 * nx - 1

    # Summed value array
    z_start = x_min + x_min
    z_final = x_max + x_max
    z = np.linspace(z_start, z_final, nz)

    # Loop through output array
    pz = np.convolve(pdf1.px, pdf2.px, mode="full")

    # Form results into PDF
    pdf_sum = PDFs.PDF(
        x=z,
        px=pz,
        **metadata.as_dict(),
    )

    return pdf_sum


def subtract_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    *,
    name: str | None = None,
    verbose: bool = False,
) -> PDFs.PDF:
    """Subtract PDF2 (Y) from PDF1 (X) to get a PDF of the difference of
    their values (Z).

    Theory:
    Subtraction of random variables is equivalent to the addition of the
    negated second variable:

        Z = X + (-Y)

    A random variable can be negated by flipping the PDF of the variable.
    Addition is carried out by convolution, as above, i.e.,

        P(Z = z) = sum(P(X = k).P(flipped_Y = z - k))

    Machinery:
    This function takes two PDFs that will be sampled on the same
    value axis.
    It creates an output array based on the input PDFs values, with the
    minimum difference being the minimum input value minus the maximum input
    value, and the maximum difference being the maximum input minus the
    minimum input.
    It then computes the probability density at each difference value by
    flipping negating the second PDF and adding it to the first.

    Parameters
    ----------
    pdf1 : PDF
        PDF from which to subtract pdf2.
    pdf2 : PDF
        PDF to subtract from pdf1.
    name : str, optional
        Name of differenced PDF.
    
    Returns
    -------
    difference_pdf : PDF
        Differenced PDF.
    """
    if verbose:
        print("Subtracting variables")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata(
        [pdf1.metadata, pdf2.metadata], name=name,
    )

    # Parameters
    x_start = pdf1.x[0]
    x_final = pdf1.x[-1]
    nx = len(pdf1)

    # Output array length
    nz = 2 * nx - 1

    # Differenced value array
    z_start = x_start - x_final
    z_final = x_final - x_start
    z = np.linspace(z_start, z_final, nz)

    # Negate variable to be subtracted
    neg_pdf2 = negate_variable(pdf2)

    # Add negated PDF2 to PDF1
    pz = np.convolve(pdf1.px, neg_pdf2.px, mode="full")

    # Form results into PDF
    pdf_diff = PDFs.PDF(
        x=z,
        px=pz,
        **metadata.as_dict(),
    )

    return pdf_diff


def multiply_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    *,
    dz: float = 0.01,
    min_product: float | None = None,
    max_product: float | None = None,
    name: str | None = None,
    variable_type: str | None = None,
    verbose: bool = False,
) -> PDFs.PDF:
    """Multiply PDF1 (X) with PDF2 (Y) to get a PDF of the product of their
    values (Z).

    Theory:
    The equation for multiplication of PDFs is similar to that for division:
    It is a weighted convolution of X and Y, with the scaling factor 1/x:

        fZ(z) = integral(fX(x).fY(z/x) 1/abs(x) dx)

    Parameters
    ----------
    pdf1 : PDF
        PDF to multiply with pdf2.
    pdf2 : PDF
        PDF to multiply with pdf1.
    dz : float, optional
        Product sample spacing.
    min_product : float, optional
        Minimum-allowable product to consider.
    max_product : float, optional
        Maximim-allowable product to consider.
    name : str, optional
        Name of product PDF.
    variable_type : str, optional
        Variable quantity.
    
    Returns
    -------
    pdf_prod : PDF
        Product PDF.
    """
    if verbose:
        print("Multiplying variables")

    # Use the four corner products to determine output array limits
    corners = [
        pdf1.x[0] * pdf2.x[0],
        pdf1.x[0] * pdf2.x[-1],
        pdf1.x[-1] * pdf2.x[0],
        pdf1.x[-1] * pdf2.x[-1],
    ]
    prod_min = np.min(corners) if min_product is None else min_product
    prod_max = np.max(corners) if max_product is None else max_product

    # Create product value array
    z = PDFs.value_arrays.precise_array(prod_min, prod_max, dz)

    # Initialize product probability density array
    n = len(z)
    pz = np.zeros(n)

    # Absolute values of pdf1
    x1_abs = np.abs(pdf1.x)

    # Non-zero index
    nonzero_ndx = (x1_abs > 10**-precision.RISER_PRECISION)

    # Non-zero values and probability densities of pdf1
    x1_nonzero = pdf1.x[nonzero_ndx]
    px1_nonzero = pdf1.px[nonzero_ndx]
    x1_abs_nonzero = x1_abs[nonzero_ndx]

    # Loop through values in the product
    for i in range(n):
        # Compute PDF2 target values (z / x)
        x2 = z[i] / x1_nonzero

        # Equivalent PDF2 density at each target value
        px2 = pdf2.pdf_at_value(x2)

        # Sum densities at product value
        pz[i] = np.sum(px1_nonzero * px2 / x1_abs_nonzero)

    # Determine product unit
    if pdf1.unit is not None and pdf2.unit is not None:
        unit = f"{pdf1.unit}.{pdf2.unit}"
    else:
        unit = None

    # Format metadata
    metadata = PDFs.PDFmetadata(
        name=name,
        variable_type=variable_type,
        unit=unit,
    )

    # Form results into PDF
    pdf_prod = PDFs.PDF(
        x=z,
        px=pz,
        **metadata.as_dict(),
    )

    return pdf_prod


def divide_variables(
    numerator: PDFs.PDF,
    denominator: PDFs.PDF,
    *,
    dz: float = 0.01,
    min_quotient: float | None = None,
    max_quotient: float | None = None,
    name: str | None = None,
    variable_type: str | None = None,
    verbose: bool = False,
) -> PDFs.PDF:
    """Divide numerator by denominator.

    Thoery:
    The equation for division of PDFs comes from Bird (2007) and later from
    Zechar and Frankel (2009):

        fV(v) = integral(fT(t).fX(x=vt).t dt)

    where v is velocity, T is time, and X is distance.
    This equation follows the same intuition for using output-side convolution
    to carry out addition and subtraction:
    For each value of the output axis, compute a weighted sum of joint
    probabilities. In this case, the distance-time joint probabilities are
    scaled by time.

    Machinery:
    Loop over the values in output array.
    An explicit nested for loop over each input variable is saved by using the
    interpolation function. Namely, the corresponding pX value to each vt
    value is interpolated along the distance (numerator) PDF. The interpolated
    numerator values can then be scaled by the corresponding time probability
    and time value, and summed directly.
    This results in slightly incrased accuracy over Zechar and Frankel's
    implementation, and greatly increased speed.

    Determining output limits:
    Unlike multiplication, numer/denom is not bilinear, so its extrema are
    only guaranteed to occur at the four corners of the input rectangle when
    the denominator's range does not straddle zero (i.e., is entirely
    positive or entirely negative). If it does straddle zero, numer/denom
    approaches +/-infinity as denom approaches zero, so no finite natural
    bound exists; min_quotient and max_quotient must be supplied explicitly
    in that case.

    Parameters
    ----------
    numerator : PDF
        Numerator distribution.
    denominator : PDF
        Denominator distribution.
    dz : float, optional
        Quotient sample spacing.
    min_quotient : float, optional
        Minimum-allowable quotient to consider. Required if denominator's
        range straddles zero.
    max_quotient : float, optional
        Maximum-allowable quotient to consider. Required if denominator's
        range straddles zero.
    name : str, optional
        Name of quotient PDF.
    variable_type : str, optional
        Variable quantity.

    Returns
    -------
    pdf_quot : PDF
        Quotient PDF.
    """
    if verbose:
        print("Dividing variables")

    # Check whether denominator's range straddles (or touches) zero
    denom_straddles_zero = (denominator.x[0] <= 0.0 <= denominator.x[-1])

    if denom_straddles_zero:
        # No finite natural bound exists: numer/denom -> +/-inf as
        # denom -> 0, so the caller must supply explicit limits
        if min_quotient is None or max_quotient is None:
            raise ValueError(
                "Denominator's range includes zero, so a natural quotient "
                "range cannot be computed. `min_quotient` and "
                "`max_quotient` must both be provided explicitly."
            )
        quot_min = min_quotient
        quot_max = max_quotient
    else:
        # Denominator is entirely positive or entirely negative, so
        # numer/denom is monotonic in each variable and its extrema over
        # the input rectangle are achieved at one of the four corners
        corners = [
            numerator.x[0] / denominator.x[0],
            numerator.x[0] / denominator.x[-1],
            numerator.x[-1] / denominator.x[0],
            numerator.x[-1] / denominator.x[-1],
        ]
        quot_min = np.min(corners) if min_quotient is None else min_quotient
        quot_max = np.max(corners) if max_quotient is None else max_quotient

    # Create quotient value array
    z = PDFs.value_arrays.precise_array(quot_min, quot_max, dz)

    # Initialize quotient probability density array
    nz = len(z)
    pz = np.zeros(nz)

    # Loop through values in quotient
    for i in range(nz):
        # Compute target numerator values (rate * denominator values)
        numer_x = z[i] * denominator.x

        # Equivalent numerator density at each target numator value
        numer_px = numerator.pdf_at_value(numer_x)

        # Sum densities at quotient value
        pz[i] = np.sum(denominator.px * numer_px * np.abs(denominator.x))

    # Determine quotient unit
    if numerator.unit is not None and denominator.unit is not None:
        unit = f"{numerator.unit}/{denominator.unit}"
    else:
        unit = None

    # Format metadata
    metadata = PDFs.PDFmetadata(
        name=name,
        variable_type=variable_type,
        unit=unit,
    )

    # Form results into PDF
    pdf_quot = PDFs.PDF(
        x=z,
        px=pz,
        **metadata.as_dict(),
    )

    return pdf_quot


# end of file
