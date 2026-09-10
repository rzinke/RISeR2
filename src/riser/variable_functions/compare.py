# -*- coding: utf-8 -*-
#
# Copyright (c) 2025-2026 Robert Zinke. Licensed under the MIT License.

"""
Functions to quantitatively compare different variables
"""


# Public API
__all__ = [
    "cosine_similarity",
    "cross_correlate_variables",
    "overlap_index",
    "ks_statistic",
]


# Import modules
import numpy as np

from .. import (
    integration,
    probability_functions as PDFs,
)


#################### COMPARISON FUNCTIONS ####################
def cosine_similarity(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    verbose: bool = False,
) -> float:
    """Compute the cosine similarity index

        r = sum[f1 f2] / sqrt[sum(f1 ^ 2) . sum(f2 ^ 2)]

    This is essentially a normalized dot product, and is analogous to the
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
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Compute coefficient
    numer = np.sum(pdf1.px * pdf2.px)
    denom = np.sqrt(np.sum(pdf1.px**2) * np.sum(pdf2.px**2))
    r = numer / denom

    # Report if requested
    if verbose:
        print(f"Cosine similarity coefficient: {r}")

    return r


def cross_correlate_variables(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cross correlation of the second variable against the first.
    Note: Unlike in classical cross correlation, which assumes infinite
    stationary signals and wraps the shifted part of the signal back around,
    this function zero-pads the second signal outside the defined portion.

    Parameters
    ----------
    pdf1 : PDF
        Reference variable to be held fixed.
    pdf2 : PDF
        Secondary variable to cross-correlate against reference.

    Returns
    -------
    lags : np.ndarray
        Lag integers.
    corr_vals : np.ndarray
        Correlation values.
    """
    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Define integer lags
    n = len(pdf1)
    lags = np.arange(-n+1, n, dtype=int)

    # Pre-allocate correlation values
    corr_vals = np.empty(2*n-1)

    # Pre-compute normalization factor for reference PDF
    ref_rss = np.sqrt(np.sum(pdf1.px**2))

    # Compute correlation values
    for i, lag in enumerate(lags):
        # Shift the secondary signal by the integer amount
        # The other way to do this would be to zero-pad the array
        px_secondary = np.roll(pdf2.px, lag)

        # Consider values outside the signal domain to be zero probability
        if lag < 0:
            px_secondary[lag:] = 0
        elif lag > 0:
            px_secondary[:lag] = 0

        # Correlation normalization factor
        norm = ref_rss * np.sqrt(np.sum(px_secondary**2))

        # Compute the correlation value
        corr_val = np.sum(pdf1.px * px_secondary)

        # Normalize correlation value
        if corr_val != 0:
            corr_val /= norm

        # Update correlation value array
        corr_vals[i] = corr_val

    return lags, corr_vals


def overlap_index(
    pdfs: list[PDFs.PDF], verbose: bool = False
) -> tuple[np.ndarray, float]:
    """Compute the overlap index for two or more PDFs.

    Pastore and Calcagni (2019)

        n(A, B) = integral(min[fA(x), fB(x)] dx)

    An alternative formulation is

        n(A, B) = 1 - (1/2 integral[ abs(fA(x) - fB(x)) dx])

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
    PDFs.metadata.check_physical_properties([pdf.metadata for pdf in pdfs])

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


def ks_statistic(
    pdf1: PDFs.PDF,
    pdf2: PDFs.PDF,
    verbose: bool = False,
) -> tuple[float, int]:
    """Compute the Kolmogorov-Smirnov statistic for two PDFs.

    The K-S statistic (D) is the largest difference between the CDFs of the
    two PDFs:

        D = sup abs(F1 - F2)

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
    # Check for consistent sampling
    PDFs.value_arrays.check_pdfs_sampling([pdf1, pdf2])

    # Warn of metadata mismatches
    PDFs.metadata.check_physical_properties([pdf1.metadata, pdf2.metadata])

    # Compute difference between CDFs
    cdf_diff = np.abs(pdf1.Px - pdf2.Px)

    # Find maximum difference
    ks_ndx = int(np.argmax(cdf_diff))
    ks_stat = cdf_diff[ks_ndx]

    # Report if requested
    if verbose:
        print(f"K-S statistic (D): {ks_stat:.2f}")

    return ks_stat, ks_ndx


# end of file
