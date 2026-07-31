# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
These functions carry out arithmetic between variables:
    addition
    subtraction
    multiplication (product distribution)
    division (ratio distribution)
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
import copy

import numpy as np

from .. import (
    probability_functions as PDFs,
    precision,
    units,
    variable_types,
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
                y[i] += x[j] * h[i - j]

    return y


#################### RANDOM VARIABLE ARITHMETIC ####################
def negate_variable(pdf: PDFs.PDF, verbose: bool = False) -> PDFs.PDF:
    """Negate a random variable expressed as a PDF.

    Negating the x-values, and flip the probability densities left for right.

    Parameters
    ----------
    pdf: PDF
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

    # Form results into PDF
    neg_pdf = PDFs.PDF(
        x=neg_x,
        px=neg_px,
        name=neg_name,
        variable_type=pdf.variable_type,
        unit=pdf.unit,
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
    For discrete PDFs, think of variable addition as a sum of joint
    probabilties as a function of values. This is exactly convolution, and is
    mathematically best expressed from the "output side".

        P(Z = z) = sum(P(X = k).P(Y = z - k))
        or
        fZ(z) = integral(fX(x).fY(z - x) dx)

    Machinery:
    This function takes two PDFs that will be sampled on the same value axis.
    It creates an output array based on the input PDFs values, with the
    minimum sum being twice the minimum input, and the maximum sum being twice
    the maximum input.
    It then computes the probability density at each summed value using output
    side convolution: that is, looping over the summed value array (iterator
    z or i) and the input value arrays (iterator k or j).

    Parameters
    ----------
    pdf1 : PDF
        PDF to add to pdf2.
    pdf2 : PDF
        PDF to add to pdf1.
    name : str
        Name of summed PDF.
    
    Returns
    -------
    sum_pdf : PDF
        Summed PDF.
    """
    if verbose:
        print("Adding variables")

    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata(
        [pdf1, pdf2], name=name, warn=True
    )

    # Parameters
    x_min = pdf1.x[0]
    x_max = pdf1.x[-1]
    dx = PDFs.value_arrays.sample_spacing_from_pdf(pdf1)
    nx = len(pdf1)

    # Output array length
    nxx = 2 * nx - 1

    # Summed value array
    xx_start = x_min + x_min
    xx_final = x_max + x_max
    xx = np.linspace(xx_start, xx_final, nxx)

    # Loop through output array
    pxx = np.convolve(pdf1.px, pdf2.px, mode="full")

    # Form results into PDF
    sum_pdf = PDFs.PDF(
        x=xx,
        px=pxx,
        normalize_area=True,
        **metadata.__dict__,
    )

    return sum_pdf


def subtract_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    *,
    limit_positive: bool = False,
    name: str = None,
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
    limit_positive : bool
        Enforce condition that values must be positive.
    name : str
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

    # Get common metadata
    metadata = PDFs.metadata.get_common_metadata(
        [pdf1, pdf2], name=name, warn=True
    )

    # Parameters
    x_start = pdf1.x[0]
    x_final = pdf1.x[-1]
    dx = PDFs.value_arrays.sample_spacing_from_pdf(pdf1)
    nx = len(pdf1)

    # Output array length
    nxx = 2 * nx - 1

    # Differenced value array
    xx_start = x_start - x_final
    xx_final = x_final - x_start
    xx = np.linspace(xx_start, xx_final, nxx)

    # Negate variable to be subtracted
    neg_pdf2 = negate_variable(pdf2)

    # Add negated PDF2 to PDF1
    pxx = np.convolve(pdf1.px, neg_pdf2.px, mode="full")

    # Enforce condition that values must be positive
    if limit_positive:
        # Keep only values > 0
        pos_ndx = (xx > 0)
        xx = xx[pos_ndx]
        pxx = pxx[pos_ndx]

    # Form results into PDF
    diff_pdf = PDFs.PDF(
        x=xx,
        px=pxx,
        normalize_area=True,
        **metadata.__dict__,
    )

    return diff_pdf


def multiply_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    *,
    dp: float = 0.01,
    min_product: float = -100.0,
    max_product: float = 100.0,
    name: str = None,
    variable_type: str | None = None,
    verbose: bool = False,
) -> PDFs.PDF:
    """Multiply PDF1 (X) with PDF2 (Y) to get a PDF of the product of their
    values (Z).

    Theory:
    The equation for multiplication of PDFs is similar to that for division:
    It is a weighted convolution of X and Y, with the scaling factor 1/x:

        fZ(z) = integral(fX(x).fY(z/x) 1/|x| dx)

    Parameters
    ----------
    pdf1 : PDF
        PDF to multiply with pdf2.
    pdf2 : PDF
        PDF to multiply with pdf1.
    dp : float
        Product sample spacing.
    min_product : float
        Minimum-allowable product to consider.
    max_product : float
        Maximim-allowable product to consider.
    name : str
        Name of product PDF.
    variable_type : str
        Variable quantity.
    
    Returns
    -------
    prod_pdf : PDF
        Product PDF.
    """
    if verbose:
        print("Multiplying variables")

    # All possible product values
    prods_all = [x1 * x2 for x1 in pdf1.x for x2 in pdf2.x]

    # Define minimum product
    prod_min = np.max([
        np.nanmin(prods_all),
        min_product
    ])

    # Determine maximum product
    prod_max = np.min([
        np.nanmax(prods_all),
        max_product
    ])

    # Create product value array
    p = PDFs.value_arrays.create_precise_value_array(prod_min, prod_max, dp)

    # Initialize product probability density array
    n = len(p)
    pp = np.zeros(n)

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
        x2 = p[i] / x1_nonzero

        # Equivalent PDF2 density at each target value
        px2 = pdf2.pdf_at_value(x2)

        # Sum densities at product value
        pp[i] = np.sum(px1_nonzero * px2 / x1_abs_nonzero)

    # Determine product unit
    if pdf1.unit is not None and pdf2.unit is not None:
        unit = f"{pdf1.unit}.{pdf2.unit}"
    else:
        unit = None

    # Form results into PDF
    prod_pdf = PDFs.PDF(
        x=p,
        px=pp,
        name=name,
        variable_type=variable_type,
        unit=unit,
        normalize_area=True,
    )

    return prod_pdf


def divide_variables(
    numerator: PDFs.PDF,
    denominator: PDFs.PDF,
    *,
    dq: float = 0.01,
    min_quotient: float = -100.0,
    max_quotient: float = 100.0,
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

    Parameters
    ----------
    numerator : PDF
        Numerator distribution.
    denominator : PDF
        Denominator distribution.
    dq : float
        Quotient sample spacing.
    min_quotient : float
        Minimum-allowable quotient to consider.
    max_quotient : float
        Maximum-allowable quotient to consider.
    name : str
        Name of quotient PDF.
    variable_type : str
        Variable quantity.
    
    Returns
    -------
    quot_pdf : PDF
        Quotient PDF.
    """
    if verbose:
        print("Dividing variables")

    # All possible quotient values
    quots_all = [
        numer / denom
        for numer in numerator.x
        for denom in denominator.x
        if denom != 0.0
    ]

    # Define minimum quotient
    quot_min = np.max([
        np.nanmin(quots_all),
        min_quotient,
    ])

    # Define maximum quotient
    quot_max = np.min([
        np.nanmax(quots_all),
        max_quotient,
    ])

    # Create quotient value array
    q = PDFs.value_arrays.create_precise_value_array(quot_min, quot_max, dq)

    # Initialize quotient probability density array
    nq = len(q)
    pq = np.zeros(nq)

    # Loop through values in quotient
    for i in range(nq):
        # Compute target numerator values (rate * denominator values)
        numer_x = q[i] * denominator.x

        # Equivalent numerator density at each target numator value
        numer_px = numerator.pdf_at_value(numer_x)

        # Sum densities at quotient value
        pq[i] = np.sum(denominator.px * numer_px * np.abs(denominator.x))

    # Determine quotient unit
    if numerator.unit is not None and denominator.unit is not None:
        unit = f"{numerator.unit}/{denominator.unit}"
    else:
        unit = None

    # Form results into PDF
    quot_pdf = PDFs.PDF(
        x=q,
        px=pq,
        name=name,
        variable_type=variable_type,
        unit=unit,
        normalize_area=True
    )

    return quot_pdf


# end of file
