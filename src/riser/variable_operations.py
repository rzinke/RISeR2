# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

"""
These functions quantify relationships or interactions between two random
variables.
"""


# Import modules
import copy

import numpy as np
import scipy as sp

from riser import precision, units
from riser.probability_functions import PDF, value_arrays


#################### GENERIC FUNCTIONS ####################
def convolve_input_side(x:np.ndarray, h:np.ndarray) -> np.ndarray:
    """Convolution operator formulated from the input side.

    Args    x, h - np.ndarray, arrays to convolve
    Returns y - np.ndarray, convolved array
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


def convolve_output_side(x:np.ndarray, h:np.ndarray) -> np.ndarray:
    """Convolution operator formulated from the output side.

    Args    x, h - np.ndarray, arrays to convolve
    Returns y - np.ndarray, convolved array
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


#################### VARIABLE COMBINATION ####################
def combine_variables(pdfs:list[PDF], verbose=False) -> PDF:
    """Compute the joint probability mass function of two or more discrete
    random variables.
    Note: Treating the PDFs as discrete greatly simplifies the calculations.

    f_X,Y(x,y) = f_X(x) * f_Y(y)

    This is similar to OxCal R_Combine.

    Args    pdfs - list[PDF], list of PDFs
    Returns joint_pdf - PDF, joint pdf
    """
    if verbose == True:
        print(f"Combining {len(pdfs)} PDFs")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling(pdfs)

    # Check units
    unit = units.check_units(pdfs)

    # Base PDF
    px = copy.deepcopy(pdfs[0].px)

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px *= pdf.px

    # Form results into PDF
    joint_pdf = PDF(pdfs[0].x, px, normalize_area=True, unit=unit)

    return joint_pdf


def merge_variables(pdfs:list[PDF], verbose=False) -> PDF:
    """Combine two or more probability mass functions by summing them

    p = f_X(x) + f_Y(y)

    and normalizing the area.

    Note that "merging" has no formal definition in the context of probability
    theory.
    This is similar to the OxCal sum function, and should not be confused with
    either compute_joint_pdf (which combines PDFs by multiplying them element-
    wise) or add_variables (which computes the sum of two independent random
    variables).
    OxCal provides a note:
    '... the 95% range for a Sum distribution give an estimate for the period
    in which 95% of the events took place not the period in which one can be
    95% sure all of the events took place.'

    Args    pdfs - list[PDF], list of PDFs to combine
    Returns merged_pdf - PDF, merged PDF
    """
    if verbose == True:
        print(f"Merging {len(pdfs)} PDFs")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling(pdfs)

    # Check units
    unit = units.check_units(pdfs)

    # Base PDF
    px = copy.deepcopy(pdfs[0].px)

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px += pdf.px

    # Form results into PDF
    merged_pdf = PDF(pdfs[0].x, px, normalize_area=True, unit=unit)

    return merged_pdf


#################### RANDOM VARIABLE ARITHMETIC ####################
def negate_variable(pdf:PDF, verbose=False) -> PDF:
    """Negate a random variable by negating the x-values, and flipping the
    probability densities left for right.

    Args    pdf, PDF to negate
    Returns neg_pdf, negated PDF
    """
    if verbose == True:
        print("Negate PDF")

    # Negate values
    neg_x = -pdf.x[::-1]

    # Flip probability densities
    neg_px = pdf.px[::-1]

    # Formulate output name
    neg_name = f"(negative) {pdf.name}" if pdf.name is not None else None

    # Form results into PDF
    args = {
        'x': neg_x,
        'px': neg_px,
        'name': neg_name,
        'variable_type': pdf.variable_type,
        'unit': pdf.unit,
    }
    neg_pdf = PDF(**args)

    return neg_pdf


def add_variables(pdf1:PDF, pdf2:PDF, name:str=None, verbose=False) -> \
        PDF:
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

    Args    pdf1, pdf2 - PDFs to add
            name - str, name of summed PDF
    Returns sum_pdf - summed PDF
    """
    if verbose == True:
        print("Adding variables")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Check units
    unit = units.check_units([pdf1, pdf2])

    # Parameters
    x_min = pdf1.x[0]
    x_max = pdf1.x[-1]
    dx = value_arrays.sample_spacing_from_pdf(pdf1)
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
    sum_pdf = PDF(xx, pxx, normalize_area=True, name=name, unit=unit)

    return sum_pdf


def subtract_variables(pdf1:PDF, pdf2:PDF, limit_positive:bool=False,
        name:str=None, verbose=False) -> PDF:
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

    Args    pdf1 - PDF from which to subtract pdf2
            pdf2 - PDF to subtract from pdf1
            limit_positive - bool, enforce condition that values must be
                positive
            name - str, name of differenced PDF
    Returns difference_pdf - differenced PDF
    """
    if verbose == True:
        print("Subtracting variables")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Check units
    unit = units.check_units([pdf1, pdf2])

    # Parameters
    x_min = pdf1.x[0]
    x_max = pdf1.x[-1]
    dx = value_arrays.sample_spacing_from_pdf(pdf1)
    nx = len(pdf1)

    # Output array length
    nxx = 2 * nx - 1

    # Differenced value array
    xx_start = x_min - x_max
    xx_final = x_max - x_min
    xx = np.linspace(xx_start, xx_final, nxx)

    # Negate variable to be subtracted
    neg_pdf2 = negate_variable(pdf2)

    # Add negated PDF2 to PDF1
    pxx = np.convolve(pdf1.px, neg_pdf2.px, mode="full")

    # Enforce condition that values must be positive
    if limit_positive == True:
        # Squash probability density of values less than zero
        pxx[xx < 0] = 0

    # Form results into PDF
    diff_pdf = PDF(xx, pxx, normalize_area=True, name=name, unit=unit)

    return diff_pdf


def divide_variables(numerator:PDF, denominator:PDF, max_quotient:float=100,
        dq:float=0.01, name:str=None, verbose=False) -> PDF:
    """Divide numerator by denominator.

    Thoery:
    The equation for division of PDFs comes from Bird (2007) and later from
    Zechar and Frankel (2009):

        fV(v) = integral(fT(t).fX(x=vt).t dt)

    where v is velocity, T is time, and X is distance.
    This equation follows the same intuition for using output-side convolution
    to carry out addition and subtraction:
    For each value of the output axis, compute something like a sum of joint
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

    Args    numerator - PDF
            denominator - PDF
            max_quotient - float, maximum-allowable quotient to consider
            dq - float, quotient sample spacing
            name - str, name of quotient PDF
    Returns quot_pdf - divided PDF
    """
    if verbose == True:
        print("Dividing variables")

    # Parameters
    n_numer = len(numerator)
    n_denom = len(denominator)

    numer_min = numerator.x[0]
    numer_max = numerator.x[-1]
    denom_min = denominator.x[0]
    denom_max = denominator.x[-1]

    # Quotient value parameters
    quot_min = numer_min / denom_max
    quot_max = np.min([max_quotient, numer_max / denom_min])

    # Create quotient value array
    q = value_arrays.create_precise_value_array(quot_min, quot_max, dq)

    # Create quotient probability density array
    nq = len(q)
    pq = np.zeros(nq)

    # Loop through values in quotient
    for i in range(nq):
        # Compute target numerator values (rate * denominator values)
        numer_x = q[i] * denominator.x

        # Equivalent numerator density at each target numator value
        numer_px = numerator.pdf_at_value(numer_x)

        # Sum numerator densities
        pq[i] = np.sum(denominator.px * numer_px * denominator.x)

    # Determine quotient unit
    unit = None
    if all([numerator.unit is not None, denominator.unit is not None]):
        unit = f"{numerator.unit}/{denominator.unit}"

    # Form results into PDF
    quot_pdf = PDF(q, pq, normalize_area=True, name=name, unit=unit)

    return quot_pdf


#################### GAP BETWEEN VARIABLES ####################
def compute_probability_between_variables(pdf1:PDF, pdf2:PDF,
            name:str=None, verbose=False) -> PDF:
    """Compute a PDF representing the domain and probability densities of
    values between two random variables.

    Theory: The probability of a value being between two uncertain values is
    equal to the probability that a value is larger than the first value
    (P(X <= x)) and smaller than the second value (1 - P(Y <= y)):

        P(X < x < Y) = CDF_X . (1 - CDF_Y) = P(X <= x) * (1 - P(Y <= y))

    Machinery: The CDFs of the first and second PDFs are pre-computed during
    PDF instantiation. Leverage these to compute the "between-PDF".

    Args    pdf1 - smaller PDF
            pdf2 - larger PDF
            name - str, name of "between" PDF
    Returns gap_pdf - PDF describing values between the two input variables
    """
    if verbose == True:
        print("Computing probability of a value between two variables.")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Check units
    unit = units.check_units([pdf1, pdf2])

    # Compute probabilities between variables
    px = pdf1.Px * (1 - pdf2.Px)

    # Form results into PDF
    gap_pdf = PDF(pdf1.x, px, normalize_area=True, name=name, unit=unit)

    return gap_pdf


#################### SIMILARITY ####################
def compute_pearson_coefficient(pdf1:PDF, pdf2:PDF, verbose=False) -> float:
    """Compute the Pearson correlation coefficient between two PDFs.

        r = sum[(xi - xbar)(yi - ybar)]
            / [ sqrt sum[(xi - xbar)^2] . sqrt sum[(yi - ybar)^2]^1/2 ]

    Because PDFs are never negative, the coefficient here is computed without
    subtracting the mean (centering).
    This is essentially a normalized dot product.

    Args    pdf1, pdf2 - PDFs to correlate
    Returns r - float, Pearson correlation coefficient
    """
    # Check for consistent sampling
    value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Check units
    unit = units.check_units([pdf1, pdf2])

    # Centered arrays
    px1_cntr = pdf1.px
    px2_cntr = pdf2.px

    # Compute coefficient
    numer = np.sum(px1_cntr * px2_cntr)
    denom = np.sqrt(np.sum(px1_cntr**2)) * np.sqrt(np.sum(px2_cntr**2))
    r = numer / denom

    # Report if requested
    if verbose == True:
        print(f"Pearson correlation coefficient: {r}")

    return r


def cross_correlate_variables(ref_pdf:PDF, sec_pdf:PDF, verbose=False) -> \
        (np.ndarray, np.ndarray):
    """Compute the cross correlation of the second variable against the first.
    Note: Unlike in classical cross correlation, which assumes infinite
    stationary signals and wraps the shifted part of the signal back around,
    this function zero-pads the second signal outside the defined portion.

    Args    ref_pdf - PDF, reference variable to be held fixed
            sec_pdf - PDF, secondary variable to cross-correlate against
                reference
    Returns lags - np.ndarray, lag integers
            corr_vals - np.ndarrays, correlation values
    """
    # Check for consistent sampling
    value_arrays.check_pdfs_sampling([ref_pdf, sec_pdf])

    # Check units
    units.check_units([ref_pdf, sec_pdf])

    # Define integer lags
    n = len(ref_pdf)
    lags = np.arange(-n+1, n, dtype=int)

    # Pre-allocate correlation values
    corr_vals = np.empty(2*n-1)

    # Pre-compute normalization factor for reference PDF
    ref_rss = np.sqrt(np.sum(ref_pdf.px**2))

    # Compute correlation values
    for i, lag in enumerate(lags):
        # Shift the secondary signal by the integer amount
        # The other way to do this would be to zero-pad the array
        px_secondary = np.roll(sec_pdf.px, lag)

        # Consider values outside the signal domain to be zero probability
        if lag < 0:
            px_secondary[lag:] = 0
        elif lag > 0:
            px_secondary[:lag] = 0

        # Correlation normalization factor
        norm = ref_rss * np.sqrt(np.sum(px_secondary**2))

        # Compute the correlation value
        corr_val = np.sum(ref_pdf.px * px_secondary)

        # Normalize correlation value
        if corr_val != 0:
            corr_val /= (ref_rss * np.sqrt(np.sum(px_secondary**2)))

        # Update correlation value array
        corr_vals[i] = corr_val

    return lags, corr_vals


def compute_overlap_index(pdfs:list[PDF], verbose=False) -> \
        (np.ndarray, float):
    """Compute the overlap index for two or more PDFs according to
    (Pastore and Calcgni, 2019):

        n(A, B) = integral(min[fA(x), fB(x)] dx)

    An alternative formulation is

        n(A, B) = 1 - (1/2 integral[ |fA(x) - fB(x)| dx])

    Args    pdfs - list[PDF], list of PDFs
    Returns eta - float, overlap metric
            px_min - np.ndarray
    """
    # Check for consistent sampling
    value_arrays.check_pdfs_sampling(pdfs)

    # Check units
    unit = units.check_units(pdfs)

    # Arrange PDFs into matrix
    pxs = np.vstack([pdf.px for pdf in pdfs])

    # Determine minimum of PDF curves
    min_ndxs = np.argmin(pxs, axis=0)
    px_min = np.array([pxs[min_ndx, i] for i, min_ndx in enumerate(min_ndxs)])

    # Integrate over overlapping region
    eta = sp.integrate.trapezoid(px_min, pdfs[0].x)

    # Report if requested
    if verbose == True:
        print(f"Overlap metric for {len(pdfs)} PDFs: {eta}")

    return px_min, eta


# end of file