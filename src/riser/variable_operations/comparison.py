# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Rob Zinke. Licensed under the MIT License.

"""
These functions are used to compare different random variables.
"""


# Public API
__all__ = [
    "compute_cosine_similarity",
    "cross_correlate_variables",
    "compute_overlap_index",
    "compute_ks_statistic",
]


# Import modules
import copy

import numpy as np

from .. import (
    integration,
    probability_functions as PDFs,
    units,
    variable_types,
)


#################### SIMILARITY METRICS ####################
def compute_cosine_similarity(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    verbose: bool = False,
) -> float:
    """Compute the cosine similarity index

        r = sum[f1 f2] / sqrt[sum(f1 ^ 2) . sum(f2 ^ 2)]

    This is essentially a normalized dot product, and is equivalent to the
    Pearson coefficient without mean-centering.
    Because PDFs are never negative, mean-centering is not necessary.

    Parameters
    ----------
    pdf1 : PDF
        PDF to correlate with pdf2.
    pdf2 : PDF
        PDF to correlate with pdf1.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    """
    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.get_common_metadata(
        [pdf1, pdf2], warn=True
    )

    # Centered arrays
    px1_cntr = pdf1.px
    px2_cntr = pdf2.px

    # Compute coefficient
    numer = np.sum(px1_cntr * px2_cntr)
    denom = np.sqrt(np.sum(px1_cntr**2) * np.sum(px2_cntr**2))
    r = numer / denom

    # Report if requested
    if verbose:
        print(f"Cosine similarity coefficient: {r}")

    return r


def cross_correlate_variables(
    ref_pdf: PDFs.PDF,
    sec_pdf: PDFs.PDF,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cross correlation of the second variable against the first.
    Note: Unlike in classical cross correlation, which assumes infinite
    stationary signals and wraps the shifted part of the signal back around,
    this function zero-pads the second signal outside the defined portion.

    Parameters
    ----------
    ref_pdf : PDF
        Reference variable to be held fixed.
    sec_pdf : PDF
        Secondary variable to cross-correlate against reference.

    Returns
    -------
    lags : np.ndarray
        Lag integers.
    corr_vals : np.ndarray
        Correlation values.
    """
    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([ref_pdf, sec_pdf])

    # Warn of metadata mismatches
    PDFs.metadata.get_common_metadata(
        [ref_pdf, sec_pdf], warn=True
    )

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


def compute_overlap_index(
    pdfs: list[PDFs.PDF], verbose: bool = False
) -> tuple[np.ndarray, float]:
    """Compute the overlap index for two or more PDFs.

    Pastore and Calcgni (2019)

        n(A, B) = integral(min[fA(x), fB(x)] dx)

    An alternative formulation is

        n(A, B) = 1 - (1/2 integral[ |fA(x) - fB(x)| dx])

    Parameters
    ----------
    pdfs : list[PDF]
        PDFs for which to compute the overlap index.

    Returns
    -------
    px_min - np.ndarray
        Minimum of PDFs at each value.
    eta : float
        Overlap metric.
    """
    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling(pdfs)

    # Warn of metadata mismatches
    PDFs.metadata.get_common_metadata(
        pdfs, warn=True
    )

    # Arrange PDFs into matrix
    pxs = np.vstack([pdf.px for pdf in pdfs])

    # Determine minimum of PDF curves
    min_ndxs = np.argmin(pxs, axis=0)
    px_min = np.array([pxs[min_ndx, i] for i, min_ndx in enumerate(min_ndxs)])

    # Integrate over overlapping region
    eta = integration.integrate(x=pdfs[0].x, px=px_min)

    # Report overlap metric
    if verbose:
        print(f"Overlap metric for {len(pdfs)} PDFs: {eta}")

    return px_min, eta


def compute_ks_statistic(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    verbose: bool = False,
) -> tuple[float, int]:
    """Compute the Komolgorov-Smirnov statistic for two PDFs.

    The K-S statistic (D) is the largest difference between the CDFs of the
    two PDFs:

        D = sup |F1 - F2|

    Parameters
    ----------
    pdf1 : PDF
        PDF to compare against pdf2.
    pdf2 : PDF
        PDF to compare against pdf1.

    Returns
    -------
    ks_stat : float
        K-S statistic.
    ks_ndx : int
        Index of K-S statistic location.
    """
    # Compute difference between CDFs
    cdf_diff = np.abs(pdf1.Px - pdf2.Px)

    # Find maximum difference
    ks_ndx = np.argmax(cdf_diff)
    ks_stat = cdf_diff[ks_ndx]

    # Report if requested
    if verbose:
        print(f"K-S statistic (D): {ks_stat:.2f}")

    return ks_stat, ks_ndx


# end of file
