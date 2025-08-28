# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Import modules
import numpy as np

from riser import precision, units
from riser.probability_functions import PDF, value_arrays


"""
In RISeR, random variables are represented as discrete PDFs.
"""


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
    px = pdfs[0].px

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
    px = pdfs[0].px

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

    Theory: For discrete PDFs, think of variable addition as a sum of joint
    probabilties as a function of values. This is exactly convolution, and is
    mathematically best expressed from the "output side".

        P(Z = z) = sum(P(X = k).P(Y = z - k))
        or
        fZ(z) = integral(fX(x).fY(z - x) dx)

    Machinery: This function takes two PDFs that will be sampled on the same
    value axis.
    It creates an output array based on the input PDFs values, with the
    minimum sum being twice the minimum input, and the maximum sum being twice
    the maximum input.
    It then computes the probability density at each summed value using output
    side convolution: that is, looping over the summed value array (iterator
    z or i) and the input value arrays (iterator k or j).

    Args    pdf1, pdf2 - PDFs to add
            name - str, name of summed PDF
    Returns sum_pdf - PDF, summed PDF
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

    Theory: Subtraction of random variables is equivalent to the addition of
    the negated second variable:

        Z = X + (-Y)

    A random variable can be negated by flipping the PDF of the variable.
    Addition is carried out by convolution, as above, i.e.,

        P(Z = z) = sum(P(X = k).P(flipped_Y = z - k))

    Machinery: This function takes two PDFs that will be sampled on the same
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
    Returns difference_pdf - PDF, differenced PDF
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


def divide_variables(numerator:PDF, denominator:PDF, verbose=False) -> PDF:
    """Divide numerator by denominator.

    Args
    Returns - PDF
    """
    if verbose == True:
        print("Dividing variables")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling(pdfs)

    return


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
def cross_correlate_variables(pdf1:PDF, pdf2:PDF):
    """
    """

    return


# end of file