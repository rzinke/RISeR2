# -*- coding: utf-8 -*-
#
# Rob Zinke
# (c) 2025 all rights reserved

# Import modules
import numpy as np

from riser.probability_functions import PDF, value_arrays


#################### VARIABLE COMBINATION ####################
def join_pdfs(pdfs:list[PDF], verbose=False) -> PDF:
    """Compute the joint probability mass function of two or more discrete
    random variables.
    Note: Treating the PDFs as discrete greatly simplifies the calculations.

    f_X,Y(x,y) = f_X(x) * f_Y(y)

    This is similar to OxCal R_Combine.

    Args    pdfs - list[PDF], list of PDFs
    Returns joint_pdf - PDF, joint pdf
    """
    if verbose == True:
        print("Computing intersection of PDFs")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling(pdfs)

    # Base PDF
    px = pdfs[0].px

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px *= pdf.px

    # Form results into PDF
    joint_pdf = PDF(pdfs[0].x, px, normalize_area=True)

    return joint_pdf


def blend_variables(pdfs:list[PDF], verbose=False) -> PDF:
    """Combine two or more probability mass functions by summing them

    p = f_X(x) + f_Y(y)

    and normalizing the area.

    This is similar to the OxCal sum function, and should not be confused with
    either compute_joint_pdf (which combines PDFs by multiplying them element-
    wise) or add_variables (which computes the sum of two independent random
    variables).
    OxCal provides a note:
    '... the 95% range for a Sum distribution give an estimate for the period
    in which 95% of the events took place not the period in which one can be
    95% sure all of the events took place.'

    Args    pdfs - list[PDF], list of PDFs to combine
    Returns blended_pdf - PDF, blended PDF
    """
    if verbose == True:
        print("Computing average of PDFs")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling(pdfs)

    # Base PDF
    px = pdfs[0].px

    # Loop through subsequent variables
    for pdf in pdfs[1:]:
        # Compute joint probability
        px += pdf.px

    # Form results into PDF
    blended_pdf = PDF(pdfs[0].x, px, normalize_area=True)

    return blended_pdf


#################### RANDOM VARIABLE ARITHMETIC ####################
def add_variables(pdf1:PDF, pdf2:PDF, verbose=False) -> PDF:
    """Add PDF1 and PDF2.

    Args    pdf1, pdf2 - PDFs to add
    Returns sum_pdf - PDF, summed PDF
    """
    if verbose == True:
        print("Adding variables")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Parameters
    x_min = pdf1.x[0]
    x_max = pdf1.x[-1]
    dx = value_arrays.sample_spacing_from_pdf(pdf1)
    nx = len(pdf1)

    # Value array
    xx_start = x_min + x_min
    xx_final = x_max + x_max
    xx = np.arange(xx_start, xx_final + dx, dx)

    # Pre-allocate output array
    nxx = 2 * nx - 1
    pxx = np.zeros(nxx)

    # Loop through output array
    for i in range(nxx):
        # Loop through points in PDF
        for j in range(nx):
            # Check in-bounds
            if (i - j >= 0) and (i - j < nx):
                pxx[i] += pdf1.px[j] * pdf2.px[i - j]

    # Form results into PDF
    sum_pdf = PDF(xx, pxx)

    return sum_pdf


def subtract_variables(pdf1:PDF, pdf2:PDF, verbose=False) -> PDF:
    """Subtract PDF2 from PDF1.

    Args
    Returns - PDF
    """
    if verbose == True:
        print("Subtracting variables")

    # Check for consistent sampling
    value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Parameters
    x_min = pdf1.x[0]
    x_max = pdf1.x[-1]
    dx = value_arrays.sample_spacing_from_pdf(pdf1)
    nx = len(pdf1)

    # Value array
    xx_start = x_min - x_max
    xx_final = x_max - x_min
    xx = np.arange(xx_start, xx_final + dx, dx)

    # Pre-allocate output array
    nxx = 2 * nx - 1
    pxx = np.zeros(nxx)

    # Loop through output array
    for i in range(nxx):
        # Loop through points in PDF
        for j in range(nx):
            break
        break

    # Form results into PDF
    diff_pdf = PDF(xx, pxx)

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


# end of file